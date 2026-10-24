import copy
import io

import pytest

from service_operator import cart


USER = "bw-cart-u"
WIRE = bytes.fromhex("0a0962772d636172742d7512070a03736b751002")
EXPECTED = {"userId": USER, "items": [{"productId": "sku", "quantity": 2}]}
PRODUCT = {"id": "sku", "name": "Telescope", "picture": "/telescope.png", "description": "A telescope",
           "categories": ["telescopes"], "priceUsd": {"currencyCode": "USD", "units": 19, "nanos": 0}}


def absent(user):
    return cart.normalize_snapshot(user, ["none", 0, -2, None])


def stored(user=USER, ttl=3599000):
    data = WIRE if user == USER else WIRE.replace(USER.encode(), user.encode())
    return cart.normalize_snapshot(user, ["hash", 1, ttl, data.hex()])


def response(*, enriched=False):
    body = copy.deepcopy(EXPECTED)
    if enriched:
        body["items"][0]["product"] = copy.deepcopy(PRODUCT)
    return {"status": 200, "body": body, "elapsed_s": 0.1, "deadline_s": 5}


def addition():
    retained = stored("bw-cart-v", 3598000)
    return dict(before=absent(USER), after=stored(), expected=copy.deepcopy(EXPECTED),
                response=response(), readback=response(enriched=True), retained_before=retained,
                retained_after={**retained, "ttl_ms": 3597000}, elapsed_s=1)


def test_independent_golden_protobuf_and_reordered_fields():
    assert cart.decode_cart(WIRE) == EXPECTED
    reordered = bytes.fromhex("120710020a03736b750a0962772d636172742d75")
    assert cart.decode_cart(reordered) == EXPECTED


@pytest.mark.parametrize("wire", [
    WIRE[:-1], b"\x0a\x80", b"\x00\x00", b"\x0a\xff\xff\xff\xff\x0f", b"\x0a\x01\xff",
    WIRE + b"\x1a\x00", WIRE + b"\x0a\x00", WIRE + WIRE[11:],
    bytes.fromhex("0a0962772d636172742d7512090a03736b7510021002"),
    bytes.fromhex("0a0962772d636172742d7512070a03736b751000"),
    bytes.fromhex("0a0962772d636172742d75120b0a03736b75108080808008"),
    b"\x80" * 10 + b"\x00", b"\x0a" + b"\xff" * 9 + b"\x02",
    b"x" * (cart.MAX_CART_BYTES + 1),
])
def test_corrupt_or_changed_schema_cannot_be_a_success(wire):
    with pytest.raises(ValueError):
        cart.decode_cart(wire)


@pytest.mark.parametrize("reply", [
    ["none", 1, -2, None], ["none", 0, -1, None], ["hash", 1, 1000, None],
    ["hash", 2, 1000, WIRE.hex()], ["hash", True, 1000, WIRE.hex()],
    ["hash", 1, True, WIRE.hex()], ["hash", 1, 1000, "a"],
    ["hash", 1, 1000, WIRE.hex().upper()], ["string", 1, 1000, WIRE.hex()],
    ["hash", 1, 1000, WIRE.replace(b"bw-cart-u", b"bw-cart-v").hex()], [],
])
def test_invalid_store_envelopes_are_evaluator_errors(reply):
    with pytest.raises(ValueError):
        cart.normalize_snapshot(USER, reply)


def test_resp_binary_payload_is_read_as_hex_without_newline_loss():
    wire = b"*4\r\n$4\r\nhash\r\n:1\r\n:3599000\r\n$40\r\n" + WIRE.hex().encode() + b"\r\n"
    assert cart.normalize_snapshot(USER, cart.read_resp(io.BytesIO(wire)))["cart"] == EXPECTED


@pytest.mark.parametrize("wire", [b"*5\r\n", b"$-2\r\n", b"$131073\r\n", b"$4\r\nabc\r\n",
                                        b"*1\r\n*1\r\n*1\r\n:1\r\n", b"-WRONGTYPE\r\n", b"?1\r\n"])
def test_truncated_oversized_or_unexpected_store_replies_fail_closed(wire):
    with pytest.raises(ValueError):
        cart.read_resp(io.BytesIO(wire))


def test_ack_and_independent_stored_cart_are_both_required():
    assert cart.check_add(**addition())["accepted"]
    data = addition()
    data["after"] = absent(USER)
    result = cart.check_add(**data)
    assert not result["accepted"]
    assert result["acknowledged"]


def test_timed_out_write_is_preserved_as_committed_without_ack():
    data = addition()
    data["response"] = {"status": None, "body": None, "elapsed_s": 5, "deadline_s": 5, "error": "TimeoutError"}
    result = cart.check_add(**data)
    assert result["exact_stored_write"] and result["committed_without_ack"]
    assert not result["accepted"]


@pytest.mark.parametrize("damage", ["old_bytes", "expired", "no_expiry", "ttl_refreshed", "read_wrong_product", "boolean_quantity", "duplicate_write", "reused_key"])
def test_false_recovery_and_unintended_state_changes_are_rejected(damage):
    data = addition()
    if damage == "old_bytes":
        data["retained_after"]["raw_hex"] = "00"
    elif damage == "expired":
        data["after"]["ttl_ms"] = 0
    elif damage == "no_expiry":
        data["after"]["ttl_ms"] = -1
    elif damage == "ttl_refreshed":
        data["retained_after"]["ttl_ms"] = 3600000
    elif damage == "read_wrong_product":
        data["readback"]["body"]["items"][0]["product"]["id"] = "other"
    elif damage == "boolean_quantity":
        data["response"]["body"]["items"][0]["quantity"] = True
    elif damage == "duplicate_write":
        data["after"]["cart"]["items"][0]["quantity"] = 4
    elif damage == "reused_key":
        data["before"] = stored()
    assert not cart.check_add(**data)["accepted"]


def test_snapshot_refuses_unowned_key_before_connecting(monkeypatch):
    def connect(*args, **kwargs):
        pytest.fail("an unowned key must not open a connection")
    monkeypatch.setattr(cart.socket, "create_connection", connect)
    with pytest.raises(ValueError, match="owned"):
        cart.snapshot("unused", "someone-elses-cart")


@pytest.mark.parametrize("field,value", [("units", True), ("units", -1), ("units", "NaN"),
                                        ("units", 2 ** 63), ("nanos", 1000000000), ("currencyCode", "EUR")])
def test_product_id_alone_cannot_hide_invalid_prices(field, value):
    product = copy.deepcopy(PRODUCT)
    product["priceUsd"][field] = value
    assert not cart.valid_product(product)


def test_proto_int64_price_units_can_be_delivered_as_decimal_text():
    product = copy.deepcopy(PRODUCT)
    product["priceUsd"]["units"] = "19"
    assert cart.valid_product(product)
