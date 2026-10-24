import json
import math
import socket


MAX_CART_BYTES = 65536
MAX_TTL_MS = 3600000
SNAPSHOT_LUA = (
    'local kind=redis.call("TYPE",KEYS[1]).ok; '
    'if kind=="none" then return {kind,0,-2,false} end; '
    'if kind~="hash" then return redis.error_reply("unexpected cart key type") end; '
    'local value=redis.call("HGET",KEYS[1],"cart"); local hex=false; '
    'if value then hex=(value:gsub(".",function(c) return string.format("%02x",string.byte(c)) end)) end; '
    'return {kind,redis.call("HLEN",KEYS[1]),redis.call("PTTL",KEYS[1]),hex}'
)


def _varint(data, offset):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        if shift == 63 and byte > 1:
            raise ValueError("protobuf varint overflow")
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
    raise ValueError("protobuf varint overflow")


def _fields(data):
    offset = 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise ValueError("invalid protobuf field")
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire == 2:
            size, offset = _varint(data, offset)
            if size > len(data) - offset:
                raise ValueError("truncated protobuf field")
            value, offset = data[offset:offset + size], offset + size
        else:
            raise ValueError("unsupported pinned cart wire type")
        yield number, wire, value


def decode_cart(data):
    if not isinstance(data, bytes) or len(data) > MAX_CART_BYTES:
        raise ValueError("cart bytes exceed the pinned contract")
    user, items, seen_user = "", [], False
    for number, wire, value in _fields(data):
        if (number, wire) == (1, 2) and not seen_user:
            user, seen_user = value.decode("utf-8"), True
        elif (number, wire) == (2, 2):
            item, seen = {"productId": "", "quantity": 0}, set()
            for field, item_wire, item_value in _fields(value):
                if field in seen:
                    raise ValueError("duplicate cart item field")
                seen.add(field)
                if (field, item_wire) == (1, 2):
                    item["productId"] = item_value.decode("utf-8")
                elif (field, item_wire) == (2, 0) and item_value < 2 ** 31:
                    item["quantity"] = item_value
                else:
                    raise ValueError("unsupported pinned cart item field")
            if not item["productId"] or item["quantity"] <= 0:
                raise ValueError("invalid stored cart item")
            items.append(item)
        else:
            raise ValueError("unsupported or duplicate pinned cart field")
    if len({item["productId"] for item in items}) != len(items):
        raise ValueError("duplicate stored product")
    return {"userId": user, "items": items}


def read_resp(stream, depth=0):
    if depth > 1:
        raise ValueError("unexpected nested store reply")
    line = stream.readline(128)
    if not line.endswith(b"\r\n"):
        raise ValueError("truncated store reply")
    kind, payload = line[:1], line[1:-2]
    if kind == b"-":
        raise ValueError("store read failed: " + payload.decode("ascii", errors="replace"))
    if kind == b"+":
        return payload.decode("ascii")
    if kind not in (b":", b"$", b"*"):
        raise ValueError("unsupported store reply")
    if not payload or not payload.removeprefix(b"-").isdigit():
        raise ValueError("invalid store reply length")
    size = int(payload)
    if kind == b":":
        return size
    if size == -1:
        return None
    if size < 0:
        raise ValueError("invalid store reply length")
    if kind == b"*":
        if size > 4:
            raise ValueError("store reply has too many fields")
        return [read_resp(stream, depth + 1) for _ in range(size)]
    if size > MAX_CART_BYTES * 2:
        raise ValueError("store reply exceeds the limit")
    data = stream.read(size + 2)
    if len(data) != size + 2 or data[-2:] != b"\r\n":
        raise ValueError("truncated store value")
    return data[:-2].decode("ascii")


def normalize_snapshot(user, reply):
    if not isinstance(reply, list) or len(reply) != 4:
        raise ValueError("unexpected cart snapshot shape")
    kind, fields, ttl, raw = reply
    if type(fields) is not int or type(ttl) is not int:
        raise ValueError("invalid cart snapshot types")
    if kind == "none":
        if (fields, ttl, raw) != (0, -2, None):
            raise ValueError("inconsistent absent cart")
        cart = None
    elif kind == "hash" and fields == 1 and isinstance(raw, str):
        if len(raw) > MAX_CART_BYTES * 2 or len(raw) % 2 or any(c not in "0123456789abcdef" for c in raw):
            raise ValueError("invalid cart hex payload")
        cart = decode_cart(bytes.fromhex(raw))
        if cart["userId"] != user:
            raise ValueError("stored cart belongs to a different user")
    else:
        raise ValueError("unexpected cart hash contents")
    return {"user": user, "key_type": kind, "hash_fields": fields,
            "ttl_ms": ttl, "raw_hex": raw, "cart": cart}


def snapshot(host, user, timeout=5):
    if not user.startswith("bw-cart-") or len(user) > 100:
        raise ValueError("store reads require an owned cart key")
    parts = [b"EVAL_RO", SNAPSHOT_LUA.encode(), b"1", user.encode()]
    wire = b"*4\r\n" + b"".join(b"$" + str(len(p)).encode() + b"\r\n" + p + b"\r\n" for p in parts)
    with socket.create_connection((host, 6379), timeout=timeout) as connection:
        connection.sendall(wire)
        with connection.makefile("rb") as stream:
            return normalize_snapshot(user, read_resp(stream))


def timely(response):
    elapsed, deadline = response.get("elapsed_s"), response.get("deadline_s")
    return (response.get("status") == 200 and not response.get("error")
            and type(elapsed) in (int, float) and type(deadline) in (int, float)
            and math.isfinite(elapsed) and math.isfinite(deadline) and 0 <= elapsed <= deadline <= 5)


def valid_product(product):
    if not isinstance(product, dict):
        return False
    if any(not isinstance(product.get(key), str) or not product[key] for key in ("id", "name", "picture")):
        return False
    if not isinstance(product.get("description"), str) or not isinstance(product.get("categories"), list):
        return False
    if any(not isinstance(category, str) for category in product["categories"]):
        return False
    price = product.get("priceUsd")
    if not isinstance(price, dict) or price.get("currencyCode") != "USD":
        return False
    units, nanos = price.get("units"), price.get("nanos")
    if isinstance(units, str) and units.isascii() and units.isdecimal() and len(units) <= 19:
        units = int(units)
    return (type(units) is int and 0 <= units < 2 ** 63
            and type(nanos) is int and 0 <= nanos < 1000000000)


def matches(response, expected, *, enriched=False):
    if not timely(response):
        return False
    body = response.get("body")
    if not isinstance(body, dict) or set(body) != {"userId", "items"} or not isinstance(body["items"], list):
        return False
    items = []
    for item in body["items"]:
        keys = {"productId", "quantity", "product"} if enriched else {"productId", "quantity"}
        if not isinstance(item, dict) or set(item) != keys:
            return False
        if enriched and (not valid_product(item["product"]) or item["product"]["id"] != item["productId"]):
            return False
        items.append({"productId": item["productId"], "quantity": item["quantity"]})
    return json.dumps({"userId": body["userId"], "items": items}, sort_keys=True) == json.dumps(expected, sort_keys=True)


def preserved(before, after, elapsed_s):
    fields = ("user", "key_type", "hash_fields", "raw_hex", "cart")
    same = all(before[key] == after[key] for key in fields)
    if before["key_type"] == "none":
        return same and before["ttl_ms"] == after["ttl_ms"] == -2
    ttl = after["ttl_ms"]
    elapsed_ok = type(elapsed_s) in (int, float) and math.isfinite(elapsed_s) and elapsed_s >= 0
    return (same and elapsed_ok and 0 < ttl <= before["ttl_ms"] <= MAX_TTL_MS
            and abs((before["ttl_ms"] - ttl) - elapsed_s * 1000) <= 5000)


def check_add(before, after, expected, response, readback, retained_before, retained_after, elapsed_s):
    if not isinstance(expected.get("userId"), str) or not expected["userId"].startswith("bw-cart-"):
        raise ValueError("expected cart must belong to this control")
    items = expected.get("items")
    if (not isinstance(items, list) or len(items) != 1 or set(items[0]) != {"productId", "quantity"}
            or not isinstance(items[0]["productId"], str) or not items[0]["productId"]
            or type(items[0]["quantity"]) is not int or not 0 < items[0]["quantity"] < 2 ** 31):
        raise ValueError("this control requires one positive cart addition")
    owned = before["user"] == after["user"] == expected["userId"]
    fresh = before["key_type"] == "none" and before["cart"] is None
    exact = after["key_type"] == "hash" and after["cart"] == expected and 0 < after["ttl_ms"] <= MAX_TTL_MS
    retained = (retained_before["user"] != expected["userId"]
                and preserved(retained_before, retained_after, elapsed_s))
    acknowledged = matches(response, expected)
    read_ok = matches(readback, expected, enriched=True)
    valid = owned and fresh and exact and retained
    return {"accepted": valid and acknowledged and read_ok, "acknowledged": acknowledged,
            "exact_stored_write": owned and fresh and exact, "retained_cart_preserved": retained,
            "readback_valid": read_ok, "committed_without_ack": valid and not acknowledged}
