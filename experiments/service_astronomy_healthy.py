import argparse
import base64
import hashlib
import http.client
import json
import os
import re
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from service_operator import cart


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(base, method, path, body, request_id):
    started = time.monotonic()
    result = {"status": None, "body": None, "deadline_s": 5,
              "request_id": request_id, "started_unix_s": time.time()}
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "X-Backworld-Request": request_id}
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(req, timeout=5) as response:
            result["status"] = response.status
            raw = response.read(262145)
            result["raw_body_b64"] = base64.b64encode(raw).decode()
            if len(raw) > 262144:
                raise ValueError("response exceeds the control limit")
            result["body"] = json.loads(raw)
    except (OSError, ValueError, http.client.HTTPException, urllib.error.URLError) as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
        if isinstance(exc, urllib.error.HTTPError):
            result["status"] = exc.code
            try:
                result["raw_body_b64"] = base64.b64encode(exc.read(262144)).decode()
            except (OSError, http.client.HTTPException) as body_error:
                result["body_error"] = type(body_error).__name__ + ": " + str(body_error)
            finally:
                exc.close()
    result["elapsed_s"] = time.monotonic() - started
    return result


def catalog_product(response, selected=None):
    products = response.get("body")
    if (not cart.timely(response) or not isinstance(products, list) or not products
            or any(not cart.valid_product(p) for p in products)
            or len({p["id"] for p in products}) != len(products)):
        raise ValueError("catalog is not a valid delivered product list")
    if selected is not None and selected not in {p["id"] for p in products}:
        raise ValueError("selected product disappeared from the catalog")
    return products[0]["id"] if selected is None else selected


def exercise(send, snapshot, record, run_id, *, interval_s=0.5, settle_s=0.25):
    if not re.fullmatch(r"[a-f0-9]{16}", run_id):
        raise ValueError("control ID must be 16 lowercase hex characters")
    users = ["bw-cart-" + run_id + suffix for suffix in ("-prior", "-current")]
    sequence = 0

    def call(method, path, body=None):
        nonlocal sequence
        sequence += 1
        request_id = run_id + f"-{sequence:03d}"
        record({"stage": "intent", "request_id": request_id, "method": method,
                "path": path, "body": body, "started_monotonic_s": time.monotonic()})
        response = send(method, path, body, request_id)
        record({"stage": "response", "request_id": request_id, "response": response,
                "finished_monotonic_s": time.monotonic()})
        return response

    def read_store(user):
        result = snapshot(user)
        record({"stage": "store", "snapshot": result, "observed_monotonic_s": time.monotonic()})
        return result

    initial = [read_store(user) for user in users]
    if any(s["key_type"] != "none" for s in initial):
        raise ValueError("owned cart IDs are not fresh; do not reset or reuse them")
    product = catalog_product(call("GET", "/api/products?currencyCode=USD"))
    verdicts = []
    for index, quantity in enumerate((2, 1)):
        user, other = users[index], users[1 - index]
        before = read_store(user)
        retained = read_store(other)
        opened = time.monotonic()
        expected = {"userId": user, "items": [{"productId": product, "quantity": quantity}]}
        response = call("POST", "/api/cart", {"userId": user, "item": expected["items"][0]})
        after, retained_after = read_store(user), read_store(other)
        elapsed = time.monotonic() - opened
        path = "/api/cart?" + urllib.parse.urlencode({"sessionId": user, "currencyCode": "USD"})
        readback = call("GET", path)
        verdict = cart.check_add(before, after, expected, response, readback, retained, retained_after, elapsed)
        time.sleep(settle_s)
        settled, retained_settled = read_store(user), read_store(other)
        verdict["stable_after_response"] = (cart.preserved(after, settled, time.monotonic() - opened - elapsed)
                                             and cart.preserved(retained, retained_settled, time.monotonic() - opened))
        verdict["accepted"] &= verdict["stable_after_response"]
        verdicts.append(verdict)
        record({"stage": "addition_verdict", "user": user, "expected": expected, "verdict": verdict})
        if not verdict["accepted"]:
            raise RuntimeError("cart addition failed verification; write will not be retried")

    before_reads = [read_store(user) for user in users]
    reads_opened = time.monotonic()
    expected = {"userId": users[1], "items": [{"productId": product, "quantity": 1}]}
    for index in range(20):
        time.sleep(max(0, reads_opened + index * interval_s - time.monotonic()))
        if index % 2:
            path = "/api/cart?" + urllib.parse.urlencode({"sessionId": users[1], "currencyCode": "USD"})
            valid = cart.matches(call("GET", path), expected, enriched=True)
        else:
            valid = catalog_product(call("GET", "/api/products?currencyCode=USD"), product) == product
        if not valid:
            raise RuntimeError("continued read was invalid")
    final = [read_store(user) for user in users]
    retained = all(cart.preserved(a, b, time.monotonic() - reads_opened) for a, b in zip(before_reads, final))
    if not retained:
        raise RuntimeError("continued reads changed or expired a cart")
    return {"passed": True, "scope": "healthy catalog/cart control only", "users": users,
            "additions": verdicts, "continued_reads": 20, "http_requests": sequence,
            "both_carts_preserved": retained, "application_or_cluster_cleanup": "external owner must verify"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"bw-astro-[a-f0-9]{10}", args.namespace):
        parser.error("use an explicitly owned bw-astro namespace")
    if not re.fullmatch(r"[a-f0-9]{16}", args.run_id):
        parser.error("run-id must be 16 lowercase hex characters")
    args.output.mkdir(parents=True, exist_ok=False)
    base = f"http://frontend-proxy.{args.namespace}.svc.cluster.local:8080"
    host = f"valkey-cart.{args.namespace}.svc.cluster.local"
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(cart.__file__))}
    manifest = {"run_id": args.run_id, "namespace": args.namespace,
        "base": base, "store": host, "source_sha256": sources, "wall_limit_s": 120,
        "request_limit": 25, "write_attempts": 2, "automatic_write_retries": 0,
        "scope": "trusted private helper in an already owned dedicated cluster; no provisioning"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def record(event):
        event["recorded_unix_s"] = time.time()
        with (args.output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(event), flush=True)

    def expired(signum, frame):
        raise RuntimeError("healthy control exceeded 120 seconds")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(120)
    try:
        record({"stage": "manifest", "manifest": manifest})
        result = exercise(lambda *a: request(base, *a), lambda user: cart.snapshot(host, user), record, args.run_id)
    except Exception as exc:
        result = {"passed": False, "error": type(exc).__name__ + ": " + str(exc), "write_retries": 0}
        record({"stage": "failed", **result})
    finally:
        signal.alarm(0)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    record({"stage": "summary", "summary": result})
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
