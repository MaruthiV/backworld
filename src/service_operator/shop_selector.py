import argparse
import base64
import hashlib
import http.client
import json
import re
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from service_operator import cart, selector_transfer


INTERVAL_S = 0.5
ACTION_S = 300
TAIL_S = 30
DRAIN_S = 5


def request(base, method, path, body, request_id):
    target = urlsplit(base)
    if target.scheme != "http" or not target.hostname or target.path or target.query or target.fragment or target.username or target.password:
        raise ValueError("source requires a fixed HTTP origin")
    payload = None if body is None else json.dumps(body).encode()
    started = time.monotonic()
    result = {"status": None, "body": None, "deadline_s": 5, "request_id": request_id}
    connection = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=4)
    try:
        connection.request(method, path, body=payload,
                           headers={"Content-Type": "application/json", "X-Backworld-Request": request_id})
        response = connection.getresponse()
        result.update(status=response.status, admission_id=response.getheader("X-Backworld-Admission"))
        raw = response.read(65537)
        result["raw_body_b64"] = base64.b64encode(raw).decode()
        if len(raw) > 65536:
            raise ValueError("source response exceeds the limit")
        result["body"] = json.loads(raw)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        connection.close()
    result["elapsed_s"] = time.monotonic() - started
    return result


def catalog_valid(response, product=None):
    products = response.get("body")
    return (cart.timely(response) and isinstance(products, list) and bool(products)
            and all(cart.valid_product(p) for p in products)
            and len({p["id"] for p in products}) == len(products)
            and (product is None or product in {p["id"] for p in products}))


class Control:
    def __init__(self, send, snapshot, record, run_id):
        if not re.fullmatch(r"[a-f0-9]{16}", run_id):
            raise ValueError("invalid private run ID")
        self.send, self.snapshot, self.record, self.run_id = send, snapshot, record, run_id
        self.lock, self.operations = threading.Lock(), threading.Lock()
        self.sequence, self.checks = 0, 0
        self.retained, self.offers, self.outcomes = {}, [], []
        self.opened = self.actor_opened = self.finished = self.offer_end = self.ended = None
        self.summary, self.product, self.thread = None, None, None
        self.prior = "bw-cart-" + run_id + "-prior"
        self.state_ok = True

    def call(self, method, path, body=None, *, phase):
        with self.lock:
            self.sequence += 1
            identity = self.run_id + f"-{self.sequence:05d}"
        self.record({"stage": "intent", "request_id": identity, "method": method, "path": path,
                     "body": body, "phase": phase, "started_s": time.monotonic()})
        response = self.send(method, path, body, identity)
        self.record({"stage": "response", "request_id": identity, "phase": phase,
                     "response": response, "finished_s": time.monotonic()})
        return response

    def read_store(self, user):
        value = self.snapshot(user)
        stamp = time.monotonic()
        self.record({"stage": "store", "snapshot": value, "observed_s": stamp})
        return value, stamp

    def preserve(self):
        for user, (before, opened) in list(self.retained.items()):
            after, ended = self.read_store(user)
            self.state_ok &= cart.preserved(before, after, ended - opened)
        return self.state_ok

    def addition(self, user, quantity, phase):
        before, _ = self.read_store(user)
        if before["key_type"] != "none":
            raise RuntimeError("cart ID is not fresh; no reset or write retry")
        responses = [self.call("GET", "/api/products?currencyCode=USD", phase=phase)]
        if not catalog_valid(responses[0], self.product):
            return {"application": "shop", "responses": responses}, False
        product = self.product or responses[0]["body"][0]["id"]
        intent = {"userId": user, "item": {"productId": product, "quantity": quantity}}
        expected = {"userId": user, "items": [intent["item"]]}
        responses.append(self.call("POST", "/api/cart", intent, phase=phase))
        after, stamp = self.read_store(user)
        responses.append(self.call("GET", "/api/cart?" + urlencode({"sessionId": user, "currencyCode": "USD"}), phase=phase))
        accepted = (cart.matches(responses[1], expected) and cart.matches(responses[2], expected, enriched=True)
                    and after["key_type"] == "hash" and after["hash_fields"] == 1 and after["cart"] == expected
                    and 0 < after["ttl_ms"] <= cart.MAX_TTL_MS)
        self.record({"stage": "addition", "user": user, "expected": expected, "accepted": accepted,
                     "response_ids": [r["request_id"] for r in responses], "snapshot": after})
        self.state_ok &= accepted
        self.retained[user] = (after, stamp)
        self.preserve()
        return {"application": "shop", "intent": intent, "responses": responses}, accepted

    def seed(self):
        if self.product is not None or self.opened is not None:
            raise RuntimeError("seed is single use")
        for attempt in range(12):
            response = self.call("GET", "/api/products?currencyCode=USD", phase="warmup")
            if catalog_valid(response):
                break
            if attempt < 11:
                time.sleep(0.5)
        else:
            raise RuntimeError("read-only warmup failed")
        self.product = response["body"][0]["id"]
        _, accepted = self.addition(self.prior, 2, "seed")
        if not accepted:
            raise RuntimeError("seed write failed; it will not be retried")
        return {"seeded": True}

    def check(self):
        if self.actor_opened is None or self.finished is not None or self.checks >= 2:
            raise RuntimeError("public workflow check is unavailable")
        self.checks += 1
        public, _ = self.addition("bw-cart-" + self.run_id + f"-check{self.checks}", 1, "public_check")
        return {**public, "responses": [{k: r[k] for k in ("status", "body", "elapsed_s", "deadline_s")}
                                        for r in public["responses"]]}

    def probe(self):
        response = self.call("GET", "/api/products?currencyCode=USD", phase="effect_probe")
        return {"valid": catalog_valid(response, self.product), "response": response}

    def start(self):
        if self.product is None or self.opened is not None:
            raise RuntimeError("source must be seeded and started exactly once")
        self.opened = time.monotonic()
        self.offer_end = self.opened + 390
        self.ended = self.offer_end + DRAIN_S
        self.record({"stage": "source_start", "opened_s": self.opened, "interval_s": INTERVAL_S,
                     "traffic_protocol_version": 2, "drain_s": DRAIN_S})
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return {"opened_s": self.opened}

    def actor(self):
        stamp = time.monotonic()
        if self.opened is None or self.actor_opened is not None or stamp - self.opened > 60:
            raise RuntimeError("actor opening is missing, repeated or late")
        self.actor_opened, self.offer_end = stamp, stamp + ACTION_S + TAIL_S
        self.ended = self.offer_end + DRAIN_S
        self.record({"stage": "actor_start", "actor_opened_s": stamp, "offer_end_s": self.offer_end, "ended_s": self.ended})
        return {"actor_opened_s": stamp, "offer_end_s": self.offer_end, "ended_s": self.ended}

    def finish(self):
        if self.actor_opened is None or self.finished is not None:
            raise RuntimeError("completion cannot be repeated")
        self.finished = time.monotonic()
        self.record({"stage": "finish", "finished_s": self.finished})
        return {"declared_complete": True}

    def offered(self, index, stamp):
        path = "/api/products?currencyCode=USD" if index % 2 == 0 else "/api/cart?" + urlencode({"sessionId": self.prior, "currencyCode": "USD"})
        response = self.call("GET", path, phase="protected")
        expected = {"userId": self.prior, "items": [{"productId": self.product, "quantity": 2}]}
        valid = catalog_valid(response, self.product) if index % 2 == 0 else cart.matches(response, expected, enriched=True)
        outcome = {"request_id": f"slot-{index}", "http_request_id": response["request_id"],
                   "finished_s": time.monotonic(), "valid": valid, "delivered": not bool(response.get("error"))}
        with self.lock:
            self.outcomes.append(outcome)
        self.record({"stage": "outcome", **outcome})

    def run(self):
        try:
            with ThreadPoolExecutor(max_workers=16) as pool:
                futures, index = [], 0
                while self.opened + index * INTERVAL_S < self.offer_end:
                    stamp = self.opened + index * INTERVAL_S
                    time.sleep(max(0, stamp - time.monotonic()))
                    offer = {"request_id": f"slot-{index}", "offered_s": stamp, "deadline_s": 5}
                    self.offers.append(offer)
                    self.record({"stage": "offer", **offer})
                    futures.append(pool.submit(self.offered, index, stamp))
                    index += 1
                time.sleep(max(0, self.ended - time.monotonic()))
                for future in futures:
                    future.result()
            with self.operations:
                public, workflow = self.addition("bw-cart-" + self.run_id + "-final", 1, "final_check")
                time.sleep(0.25)
                state = self.preserve()
            loss = selector_transfer.workload_loss(self.offers, self.outcomes, self.opened, self.ended)
            indexed = {o["request_id"]: o for o in self.outcomes}
            tail = [o for o in self.offers if o["offered_s"] >= self.offer_end - 10]
            recovered = len(tail) >= 20 and all(indexed[o["request_id"]]["valid"] is True
                and indexed[o["request_id"]]["delivered"] is True
                and indexed[o["request_id"]]["finished_s"] <= min(self.ended, o["offered_s"] + 5) for o in tail)
            post = sum(o["offered_s"] >= self.finished for o in self.offers) if self.finished is not None else 0
            summary = {"loss": loss, "workflow_valid": workflow and selector_transfer.public_workflow_valid(public),
                            "state_preserved": state, "recovered": recovered, "post_completion": post,
                            "source_on_time": self.finished is not None and self.actor_opened is not None
                            and 0 <= self.finished - self.actor_opened < ACTION_S,
                            "opened_s": self.opened, "actor_opened_s": self.actor_opened,
                            "finished_s": self.finished, "offer_end_s": self.offer_end, "ended_s": self.ended}
        except Exception as exc:
            summary = {"error": type(exc).__name__ + ": " + str(exc)}
        self.record({"stage": "summary", "summary": summary})
        self.summary = summary


def handler(control):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Length", "0") != "0":
                self.send_error(400)
                return
            try:
                if self.path == "/status":
                    value = {"ready": True, "summary": control.summary}
                else:
                    operation = {"/seed": control.seed, "/start": control.start, "/probe": control.probe,
                                 "/actor": control.actor, "/check": control.check, "/finish": control.finish}[self.path]
                    with control.operations:
                        value = operation()
                body, status = json.dumps(value, allow_nan=False).encode(), 200
            except Exception as exc:
                control.record({"stage": "rpc_error", "path": self.path, "error": type(exc).__name__ + ": " + str(exc)})
                body, status = b'{"error":"private operation failed; do not retry mutations"}', 500
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"bw-astro-[a-f0-9]{10}", args.namespace):
        parser.error("owned shop namespace required")
    lock = threading.Lock()

    def record(event):
        with lock:
            print(json.dumps(event, allow_nan=False), flush=True)

    record({"stage": "manifest", "run_id": args.run_id, "namespace": args.namespace,
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              (Path(__file__), Path(cart.__file__), Path(selector_transfer.__file__))},
            "interval_s": INTERVAL_S, "action_s": ACTION_S, "tail_s": TAIL_S, "drain_s": DRAIN_S,
            "traffic_protocol_version": 2, "write_retries": 0})
    signal.alarm(570)
    control = Control(lambda *a: request("http://127.0.0.1:8080", *a),
                      lambda user: cart.snapshot(f"valkey-cart.{args.namespace}.svc.cluster.local", user), record, args.run_id)
    ThreadingHTTPServer(("0.0.0.0", 8081), handler(control)).serve_forever()


if __name__ == "__main__":
    main()
