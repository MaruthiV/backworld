import copy

import pytest

from service_operator.workflow import check_booking


PRIOR = {"_id": "old", "hotelId": "4", "customerName": "Alice",
         "inDate": "2015-04-09", "outDate": "2015-04-10", "number": 1}
EXPECTED = {"hotelId": "1", "customerName": "new-booking",
            "inDate": "2026-11-01", "outDate": "2026-11-02", "number": 1}
ACK = {"status": 200, "body": {"message": "Reserve successfully!"},
       "elapsed_s": 0.1, "deadline_s": 5}


def rows():
    return [copy.deepcopy(PRIOR), {"_id": "new", **EXPECTED}]


def test_acknowledged_booking_preserves_prior_record():
    result = check_booking([PRIOR], rows(), EXPECTED, ACK)
    assert result["accepted"]
    assert result["outcome"] == "confirmed"


def test_success_body_without_write_is_rejected():
    result = check_booking([PRIOR], [PRIOR], EXPECTED, ACK)
    assert not result["accepted"]
    assert result["outcome"] == "ack_without_valid_write"


def test_duplicate_booking_is_rejected():
    after = rows() + [{"_id": "duplicate", **EXPECTED}]
    result = check_booking([PRIOR], after, EXPECTED, ACK)
    assert not result["accepted"]
    assert result["inserted_count"] == 2


@pytest.mark.parametrize("field,value", [
    ("hotelId", "2"), ("customerName", "someone-else"),
    ("inDate", "2026-11-02"), ("outDate", "2026-11-03"),
    ("number", 2), ("number", True), ("extra", "unexpected"),
])
def test_wrong_booking_fields_are_rejected(field, value):
    after = rows()
    after[1][field] = value
    assert not check_booking([PRIOR], after, EXPECTED, ACK)["accepted"]


@pytest.mark.parametrize("change", ["delete", "modify", "replace_id"])
def test_prior_booking_must_survive_unchanged(change):
    after = rows()
    if change == "delete":
        after.pop(0)
    elif change == "modify":
        after[0]["number"] = 2
    else:
        after[0]["_id"] = "replacement"
    result = check_booking([PRIOR], after, EXPECTED, ACK)
    assert not result["accepted"]
    assert result["outcome"] == "prior_state_changed"


@pytest.mark.parametrize("change", [
    {"error": "TimeoutError", "status": None, "body": None},
    {"elapsed_s": 6}, {"status": 500}, {"body": {"message": "ok"}},
])
def test_write_without_timely_acknowledgement_is_not_success(change):
    result = check_booking([PRIOR], rows(), EXPECTED, {**ACK, **change})
    assert not result["accepted"]
    assert result["exact_new_write"]
    assert result["outcome"] == "committed_without_ack"


def test_unrelated_write_is_rejected():
    after = rows() + [{**PRIOR, "_id": "stray", "customerName": "stray"}]
    assert not check_booking([PRIOR], after, EXPECTED, ACK)["accepted"]


def test_malformed_snapshot_is_an_evaluator_error():
    with pytest.raises(ValueError, match="duplicate"):
        check_booking([PRIOR], rows() + [PRIOR], EXPECTED, ACK)
