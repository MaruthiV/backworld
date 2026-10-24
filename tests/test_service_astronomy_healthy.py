import copy
import importlib.util
import urllib.parse
from pathlib import Path

import pytest


path = Path(__file__).parents[1] / "experiments/service_astronomy_healthy.py"
spec = importlib.util.spec_from_file_location("astronomy_healthy", path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
PRODUCT = {"id": "sku", "name": "Telescope", "picture": "/telescope.png", "description": "A telescope",
           "categories": ["telescopes"], "priceUsd": {"currencyCode": "USD", "units": 19, "nanos": 0}}


def ack(body):
    return {"status": 200, "body": body, "elapsed_s": 0.01, "deadline_s": 5}


def test_uncertain_post_is_never_retried_and_stored_outcome_is_recorded():
    calls, events, records = [], [], {}

    def snapshot(user):
        value = records.get(user)
        return {"user": user, "key_type": "hash" if value else "none", "hash_fields": 1 if value else 0,
                "ttl_ms": 3599000 if value else -2, "raw_hex": "fixture" if value else None, "cart": value}

    def send(method, path, body, request_id):
        calls.append((method, path))
        if path.startswith("/api/products"):
            return ack([copy.deepcopy(PRODUCT)])
        if method == "POST":
            records[body["userId"]] = {"userId": body["userId"], "items": [body["item"]]}
            return {"status": None, "body": None, "elapsed_s": 5, "deadline_s": 5, "error": "TimeoutError"}
        user = next(iter(records))
        value = copy.deepcopy(records[user])
        value["items"][0]["product"] = copy.deepcopy(PRODUCT)
        return ack(value)

    with pytest.raises(RuntimeError, match="not be retried"):
        probe.exercise(send, snapshot, events.append, "0123456789abcdef", interval_s=0, settle_s=0)
    assert sum(method == "POST" for method, _ in calls) == 1
    verdict = next(e["verdict"] for e in events if e["stage"] == "addition_verdict")
    assert verdict["committed_without_ack"] and not verdict["accepted"]
    assert events[0]["stage"] == "store"
    intent_index = next(i for i,e in enumerate(events) if e["stage"] == "intent" and e["method"] == "POST")
    response_index = next(i for i,e in enumerate(events) if e["stage"] == "response" and e["response"].get("error"))
    assert intent_index < response_index


def test_reused_ids_stop_before_any_http_request():
    def send(*args):
        pytest.fail("a nonfresh control must not send requests")
    with pytest.raises(ValueError, match="not fresh"):
        probe.exercise(send, lambda user: {"key_type": "hash"}, lambda event: None,
                       "0123456789abcdef", interval_s=0, settle_s=0)


def test_catalog_order_can_change_without_changing_the_selected_product():
    products = [{**copy.deepcopy(PRODUCT), "id": key} for key in ("a", "b")]
    assert probe.catalog_product(ack(products), "b") == "b"
    with pytest.raises(ValueError, match="disappeared"):
        probe.catalog_product(ack(products), "missing")


def test_complete_control_accounts_for_two_writes_and_twenty_continued_reads():
    calls, events, records = [], [], {}

    def snapshot(user):
        value = copy.deepcopy(records.get(user))
        return {"user": user, "key_type": "hash" if value else "none", "hash_fields": 1 if value else 0,
                "ttl_ms": 3599000 if value else -2, "raw_hex": "fixture" if value else None, "cart": value}

    def send(method, path, body, request_id):
        calls.append((method, path, request_id))
        if path.startswith("/api/products"):
            return ack([copy.deepcopy(PRODUCT)])
        if method == "POST":
            assert body["userId"] not in records
            records[body["userId"]] = {"userId": body["userId"], "items": [copy.deepcopy(body["item"])]}
            return ack(copy.deepcopy(records[body["userId"]]))
        user = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)["sessionId"][0]
        value = copy.deepcopy(records[user])
        value["items"][0]["product"] = copy.deepcopy(PRODUCT)
        return ack(value)

    result = probe.exercise(send, snapshot, events.append, "0123456789abcdef", interval_s=0, settle_s=0)
    assert result["passed"] and result["both_carts_preserved"]
    assert result["continued_reads"] == 20
    assert len(calls) == result["http_requests"] == 25
    assert len({request_id for _,_,request_id in calls}) == 25
    assert sum(method == "POST" for method,_,_ in calls) == 2
    assert sorted(v["items"][0]["quantity"] for v in records.values()) == [1, 2]
