import json


BOOKING_MESSAGE = {"message": "Reserve successfully!"}


def check_window(before, immediate, after, events):
    old = {row["_id"]: row for row in before["reservations"]}
    new = {row["_id"]: row for row in after["reservations"]}
    prior_ok = all(json.dumps(new.get(key), sort_keys=True) == json.dumps(row, sort_keys=True) for key, row in old.items())
    added = [row for key, row in new.items() if key not in old]
    expected = {event["expected"]["customerName"]: event["expected"] for event in events}
    accounted = len(expected) == len(events) and all(row.get("customerName") in expected for row in added)
    unique_ids = len(old) == len(before["reservations"]) and len(new) == len(after["reservations"])
    invariant = (prior_ok and json.dumps(before["capacity"], sort_keys=True) == json.dumps(after["capacity"], sort_keys=True)
                 and json.dumps(immediate, sort_keys=True) == json.dumps(after, sort_keys=True) and accounted and unique_ids)
    verdicts = []
    for event in events:
        rows = [row for row in added if row.get("customerName") == event["expected"]["customerName"]]
        verdict = check_booking([], rows, event["expected"], event["response"])
        invariant &= (len(rows) == 0 or verdict["exact_new_write"]) and (not verdict["acknowledged"] or verdict["exact_new_write"])
        verdicts.append(verdict)
    for event, verdict in zip(events, verdicts):
        on_time = event["acknowledgement_monotonic_s"] - event["offered_monotonic_s"] <= 2
        verdict.update({"prior_preserved": prior_ok, "invariant_ok": invariant, "on_time": on_time,
                        "accepted": verdict["accepted"] and invariant and on_time})
    return verdicts


def check_booking(before, after, expected, response):
    def index(rows):
        if any(not isinstance(row.get("_id"), str) for row in rows):
            raise ValueError("reservation IDs must be strings")
        indexed = {row["_id"]: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError("duplicate reservation IDs in snapshot")
        return indexed

    old, new = index(before), index(after)
    preserved = all(json.dumps(new.get(key), sort_keys=True) == json.dumps(row, sort_keys=True)
                    for key, row in old.items())
    inserted = [row for key, row in new.items() if key not in old]
    payloads = [{key: value for key, value in row.items() if key != "_id"}
                for row in inserted]
    exact_write = (
        len(payloads) == 1
        and json.dumps(payloads[0], sort_keys=True) == json.dumps(expected, sort_keys=True)
    )
    acknowledged = (
        response.get("status") == 200
        and response.get("body") == BOOKING_MESSAGE
        and not response.get("error")
        and 0 <= response["elapsed_s"] <= response["deadline_s"]
    )
    valid_state = preserved and exact_write
    if not preserved:
        outcome = "prior_state_changed"
    elif valid_state:
        outcome = "confirmed" if acknowledged else "committed_without_ack"
    else:
        outcome = "ack_without_valid_write" if acknowledged else "not_confirmed"
    return {
        "accepted": acknowledged and valid_state,
        "outcome": outcome,
        "acknowledged": acknowledged,
        "prior_preserved": preserved,
        "exact_new_write": exact_write,
        "inserted_count": len(inserted),
    }
