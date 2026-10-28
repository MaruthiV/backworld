import base64
import copy
import hashlib
import http.client
import importlib.util
import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs

import pytest

from service_operator import astronomy_host, relay, selector_native, selector_transfer, shop_selector, tools


RUN = "0123456789abcdef"
PRODUCT = {"id": "sku", "name": "Telescope", "picture": "/image.png", "description": "A telescope",
           "categories": ["telescopes"], "priceUsd": {"currencyCode": "USD", "units": 19, "nanos": 0}}


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class InlinePool:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, *args):
        function(*args)
        return SimpleNamespace(result=lambda: None)


class Shop:
    def __init__(self, clock):
        self.clock, self.carts, self.relay = clock, {}, []
        self.broken = self.unadmitted = self.uncertain_write = False
        self.writes = []

    def snapshot(self, user):
        if user not in self.carts:
            return {"user": user, "key_type": "none", "hash_fields": 0, "ttl_ms": -2, "raw_hex": None, "cart": None}
        value, opened = self.carts[user]
        return {"user": user, "key_type": "hash", "hash_fields": 1,
                "ttl_ms": 3600000 - round((self.clock.now - opened) * 1000), "cart": copy.deepcopy(value),
                "raw_hex": astronomy_host.expected_wire(user, value["items"][0]["productId"], value["items"][0]["quantity"])}

    def send(self, method, path, body, identity):
        self.clock.now += 0.01
        if self.unadmitted:
            return {"status": None, "body": None, "error": "ConnectionRefusedError", "elapsed_s": 0.01,
                    "deadline_s": 5, "request_id": identity}
        admission = {"id": "admission-" + identity, "client_request_id": identity, "method": method,
                     "request_hash": hashlib.sha256(path.encode()).hexdigest(),
                     "request_body_sha256": hashlib.sha256(b"" if body is None else json.dumps(body).encode()).hexdigest()}
        status, value = 200, None
        if self.broken:
            status, value = 503, {"error": "no upstream"}
        elif method == "POST":
            self.writes.append(copy.deepcopy(body))
            user = body["userId"]
            value = {"userId": user, "items": [copy.deepcopy(body["item"])]}
            self.carts[user] = copy.deepcopy(value), self.clock.now
        elif path.startswith("/api/products"):
            value = [PRODUCT]
        else:
            query = shop_selector.urlsplit(path).query
            user = parse_qs(query)["sessionId"][0]
            value = copy.deepcopy(self.carts[user][0])
            for item in value["items"]:
                item["product"] = PRODUCT
        raw = json.dumps(value).encode()
        self.relay += [{"stage": "started", **admission}, {"stage": "completed", **admission,
                       "status": status, "delivered": True, "body_sha256": hashlib.sha256(raw).hexdigest()}]
        if self.uncertain_write and method == "POST":
            return {"status": None, "body": None, "error": "RemoteDisconnected", "elapsed_s": 0.01,
                    "deadline_s": 5, "request_id": identity}
        return {"status": status, "body": copy.deepcopy(value), "raw_body_b64": base64.b64encode(raw).decode(),
                "admission_id": admission["id"], "elapsed_s": 0.01, "deadline_s": 5, "request_id": identity}


@pytest.fixture
def environment(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(shop_selector, "time", clock)
    monkeypatch.setattr(shop_selector, "ThreadPoolExecutor", InlinePool)
    monkeypatch.setattr(shop_selector, "threading", SimpleNamespace(Lock=__import__("threading").Lock,
                        Thread=lambda **kwargs: SimpleNamespace(start=lambda: None)))
    shop, events = Shop(clock), []
    control = shop_selector.Control(shop.send, shop.snapshot, lambda row: events.append(copy.deepcopy(row)), RUN)
    return clock, shop, control, events


def complete(environment, condition="healthy", *, unadmitted=False, actor_delay_s=0):
    clock, shop, control, events = environment
    control.seed()
    control.start()
    shop.broken = condition != "healthy"
    if shop.broken:
        assert not control.probe()["valid"]
        assert not control.probe()["valid"]
    clock.sleep(actor_delay_s)
    control.actor()
    if condition == "known_repair":
        shop.broken = False
    if condition in ("healthy", "known_repair"):
        public = control.check()
        assert selector_transfer.public_workflow_valid(public)
    control.finish()
    shop.unadmitted = unadmitted
    control.run()
    return selector_native.audit(events, shop.relay, RUN)


@pytest.mark.parametrize("condition", selector_transfer.CONDITIONS)
def test_source_full_horizon_and_independent_raw_audit(environment, condition):
    review = complete(environment, condition)
    expected = condition in ("healthy", "known_repair")
    assert review["workflow_valid"] is expected and review["recovered"] is expected
    assert review["state_preserved"] and review["source_on_time"]
    assert review["loss"]["offered"] >= 660 and review["post_completion"] >= 650
    assert review["offer_end_s"] - review["actor_opened_s"] == 330
    assert review["ended_s"] - review["offer_end_s"] == 5
    assert review["loss"]["failed"] == (0 if expected else review["loss"]["offered"])
    assert review["exact_writes"] == (3 if expected else 1)


def test_final_offer_keeps_its_full_deadline_after_admission_closes(environment):
    review = complete(environment, actor_delay_s=0.002670626)
    _, _, control, _ = environment
    last_offer = control.offers[-1]
    last_outcome = next(o for o in control.outcomes if o["request_id"] == last_offer["request_id"])
    assert 0 < review["offer_end_s"] - last_offer["offered_s"] < 0.003
    assert review["offer_end_s"] < last_outcome["finished_s"] <= last_offer["offered_s"] + 5 <= review["ended_s"]
    assert review["loss"]["offered"] == 661 and review["loss"]["failed"] == 0
    assert review["recovered"]


@pytest.mark.parametrize("damage", ["slot", "duplicate_slot", "raw", "wire", "ttl", "relay", "retry", "false_valid", "false_summary", "schedule", "horizon"])
def test_independent_audit_rejects_corrupt_or_missing_evidence(environment, damage):
    complete(environment)
    _, shop, _, events = environment
    if damage == "slot":
        events.remove(next(e for e in events if e["stage"] == "offer"))
    elif damage == "duplicate_slot":
        events.append(copy.deepcopy(next(e for e in events if e["stage"] == "offer")))
    elif damage == "raw":
        next(e for e in events if e["stage"] == "response")["response"]["body"] = []
    elif damage == "wire":
        next(e for e in events if e["stage"] == "addition")["snapshot"]["raw_hex"] = "00"
    elif damage == "ttl":
        [e for e in events if e["stage"] == "store" and e["snapshot"]["user"].endswith("-prior")][-1]["snapshot"]["ttl_ms"] = 3600000
    elif damage == "relay":
        shop.relay.pop()
    elif damage == "retry":
        events.append(copy.deepcopy(next(e for e in events if e["stage"] == "intent" and e["method"] == "POST")))
    elif damage == "false_valid":
        next(e for e in events if e["stage"] == "outcome")["valid"] = False
    elif damage == "false_summary":
        next(e for e in events if e["stage"] == "summary")["summary"]["loss"]["valid"] -= 1
    elif damage == "schedule":
        next(e for e in events if e["stage"] == "offer")["offered_s"] += 0.1
    else:
        next(e for e in events if e["stage"] == "actor_start")["ended_s"] -= 1
    with pytest.raises(ValueError):
        selector_native.audit(events, shop.relay, RUN)


def test_unadmitted_transport_failures_still_count_as_source_loss(environment):
    review = complete(environment, "premature_finish", unadmitted=True)
    assert review["loss"]["failed"] == review["loss"]["offered"]
    assert review["unadmitted_failed_attempts"] >= review["loss"]["offered"]


def test_uncertain_post_is_never_retried(environment):
    _, shop, control, events = environment
    control.seed()
    control.start()
    control.actor()
    shop.uncertain_write = True
    public = control.check()
    assert not selector_transfer.public_workflow_valid(public)
    assert len(shop.writes) == 2
    assert not control.state_ok
    assert "snapshot" not in json.dumps(public) and "accepted" not in json.dumps(public)


def test_public_checks_use_new_ids_and_reject_third_or_after_finish(environment):
    _, shop, control, _ = environment
    control.seed()
    control.start()
    control.actor()
    first, second = control.check(), control.check()
    assert first["intent"]["userId"] != second["intent"]["userId"] != control.prior
    assert set(first) == {"application", "responses", "intent"}
    assert all(set(r) == {"status", "body", "elapsed_s", "deadline_s"} for r in first["responses"])
    with pytest.raises(RuntimeError):
        control.check()
    control.finish()
    with pytest.raises(RuntimeError):
        control.check()
    assert len(shop.writes) == 3


@contextmanager
def server(handler):
    import threading
    instance = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance.server_address[1]
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


def test_actual_shop_relay_post_and_admission_match_source_bytes():
    posts, events = [], []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            posts.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(posts[-1])

        def log_message(self, *args):
            pass

    intent = {"userId": "bw-cart-owned", "item": {"productId": "sku", "quantity": 1}}
    with server(Upstream) as upstream, server(relay.handler(f"http://127.0.0.1:{upstream}", events.append, "shop")) as port:
        response = shop_selector.request(f"http://127.0.0.1:{port}", "POST", "/api/cart", intent, "owned-1")
        assert response["status"] == 200 and response["body"] == intent
        assert response["admission_id"] == events[0]["id"]
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        for method, path, body in [("POST", "/api/products", b"{}"), ("GET", "/hotels", None),
                                   ("POST", "/api/cart", b"x" * 4097), ("GET", "http://elsewhere/api/cart", None)]:
            connection.request(method, path, body=body)
            rejected = connection.getresponse()
            assert rejected.status in (404, 413)
            rejected.read()
        connection.close()
    assert posts == [json.dumps(intent).encode()]
    assert len(events) == 2 and events[-1]["delivered"]


def test_actual_relay_does_not_retry_uncertain_post():
    writes, events = [], []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            writes.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.close_connection = True

        def log_message(self, *args):
            pass

    with server(Upstream) as upstream, server(relay.handler(f"http://127.0.0.1:{upstream}", events.append, "shop")) as port:
        response = shop_selector.request(f"http://127.0.0.1:{port}", "POST", "/api/cart", {"userId": "bw-cart-owned"}, "owned-1")
    assert response["status"] == 502 and len(writes) == 1
    assert events[-1]["error"] == "RemoteDisconnected"


def test_actual_private_rpc_limits_public_data_and_rejects_body(environment):
    _, _, control, _ = environment
    control.seed()
    control.start()
    control.actor()
    with server(shop_selector.handler(control)) as port:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.request("POST", "/check", body=b"")
        response = connection.getresponse()
        body = json.loads(response.read())
        assert response.status == 200 and selector_transfer.public_workflow_valid(body)
        assert "snapshot" not in json.dumps(body) and "grade" not in json.dumps(body)
        connection.request("POST", "/check", body=b"{}")
        rejected = connection.getresponse()
        assert rejected.status == 400
        rejected.read()
        connection.close()


def test_package_rejects_changed_inputs_before_creating_output(tmp_path):
    path = Path(__file__).parents[1] / "experiments/service_shop_selector.py"
    spec = importlib.util.spec_from_file_location("shop_package", path)
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "bundle.json").write_text(json.dumps({"files_sha256": {"application.json": "wrong"}}))
    (inputs / "application.json").write_text("{}")
    with pytest.raises(ValueError, match="input bundle file changed"):
        package.prepare(inputs, tmp_path / "injector", tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("condition", [*selector_transfer.CONDITIONS, "ineffective_fault"])
def test_native_cycle_wires_public_tools_fault_gate_audit_and_tunnel_cleanup(environment, monkeypatch, tmp_path, condition):
    actual = condition if condition != "ineffective_fault" else "known_repair"
    complete(environment, actual)
    clock, shop, control, events = environment
    namespace, private = "bw-astro-0123456789", "bw-check-0123456789"
    digest = "sha256:" + "a" * 64
    pod = {"metadata": {"name": "helper", "uid": "helper-uid", "labels": {"backworld.run": RUN},
                        "ownerReferences": [{"kind": "Job", "uid": "job-uid", "controller": True}]},
           "spec": {"automountServiceAccountToken": False, "containers": [{"name": name, "image": "pinned"} for name in ("source", "relay")]},
           "status": {"phase": "Running", "conditions": [{"type": "Ready", "status": "True"}],
                      "containerStatuses": [{"name": name, "imageID": "pinned@" + digest, "containerID": name,
                                             "restartCount": 0, "state": {"running": {}}} for name in ("source", "relay")]}}
    job_plan = {"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": "selector-control", "namespace": private, "labels": {"backworld.run": RUN}}}
    owner = SimpleNamespace(objects=[job_plan], get=lambda obj: {**obj, "metadata": {**obj["metadata"], "uid": "job-uid"}},
                            receipts={astronomy_host.identity(job_plan)["key"]: {"uid": "job-uid"}})
    service = {"metadata": {"name": "frontend", "uid": "service-uid", "resourceVersion": "1"},
               "spec": {"selector": {"app": "frontend"}, "ports": [{"port": 8080}]}}
    before = {"services": [copy.deepcopy(service)]}
    manifests = {"source_hashes": {"shop_selector.py": "fixture"}, "run_id": RUN,
                 "application_namespace": namespace, "private_namespace": private}
    events.insert(0, {"stage": "manifest", "run_id": RUN, "namespace": namespace, "source_sha256": manifests["source_hashes"],
                      "traffic_protocol_version": 2, "drain_s": 5})
    calls, patches, rpc_paths = [], [], []

    class Command:
        env, deadline = {}, clock.now + 1000

        def __call__(self, args, **kwargs):
            calls.append(args)
            if "auth" in args:
                return "yes\n" if args[args.index("can-i") + 2] in ("services/frontend", "deployments.apps/frontend") else "no\n"
            if "logs" in args:
                rows = events if args[-1] == "source" else shop.relay
                return "\n".join(json.dumps(row) for row in rows)
            if "patch" in args:
                resource = args[args.index("patch") + 1]
                patch = json.loads(args[-1])
                patches.append((resource, patch))
                if resource == "service":
                    service["spec"]["selector"] = patch[-1]["value"]
                return "{}"
            resource = args[args.index("get") + 1]
            if resource == "pod" or resource == "pods" and "-l" not in args:
                return json.dumps(pod if resource == "pod" else {"items": [pod]})
            if resource == "service":
                return json.dumps(service)
            if resource == "deployment":
                return json.dumps({"metadata": {"uid": "frontend-uid", "generation": 1, "resourceVersion": "1"},
                                   "spec": {"replicas": 1, "selector": {"matchLabels": {"app": "frontend"}},
                                            "template": {"metadata": {"labels": {"app": "frontend"}}}},
                                   "status": {"observedGeneration": 1, **{key: 1 for key in ("replicas", "updatedReplicas", "readyReplicas", "availableReplicas")}}})
            if resource == "pods":
                return json.dumps({"items": [{"metadata": {"name": "frontend", "uid": "frontend-pod"},
                                             "status": {"conditions": [{"type": "Ready", "status": "True"}]}}]})
            if resource == "endpointslices":
                endpoints = [] if "current_service_name" in service["spec"]["selector"] else [{"targetRef": {"uid": "frontend-pod"}, "conditions": {"ready": True}}]
                return json.dumps({"items": [{"endpoints": endpoints}]})
            raise AssertionError(args)

    class Tunnel:
        pid, returncode = 1234, None

        def __init__(self, args, **kwargs):
            kwargs["stdout"].write("Forwarding from 127.0.0.1:12345 -> 8081\n")
            kwargs["stdout"].flush()

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, **kwargs):
            return self.returncode

    class RPC:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, method, path, **kwargs):
            rpc_paths.append(path)
            self.path = path

        def getresponse(self):
            value = {}
            if self.path == "/probe":
                value = {"valid": condition == "ineffective_fault" or rpc_paths.count("/probe") == 1,
                         "response": {"admission_id": "admitted"}}
            elif self.path == "/status":
                value = {"summary": control.summary}
            elif self.path == "/check":
                addition = next(e for e in events if e["stage"] == "addition" and e["user"].endswith("-check1"))
                responses = {e["request_id"]: e["response"] for e in events if e["stage"] == "response"}
                value = {"application": "shop", "intent": {"userId": addition["user"], "item": addition["expected"]["items"][0]},
                         "responses": [{k: responses[i][k] for k in ("status", "body", "elapsed_s", "deadline_s")} for i in addition["response_ids"]]}
            return SimpleNamespace(status=200, read=lambda n: json.dumps(value).encode())

        def close(self):
            pass

    def inject(target, names):
        value = target.kubectl.get_service_json("frontend", namespace)
        value["spec"]["selector"]["current_service_name"] = "frontend"
        target.kubectl.patch_service("frontend", namespace, value)

    monkeypatch.setattr(selector_native, "time", clock)
    monkeypatch.setattr(tools, "time", clock)
    monkeypatch.setattr(selector_native.subprocess, "Popen", Tunnel)
    monkeypatch.setattr(selector_native.http.client, "HTTPConnection", RPC)
    monkeypatch.setattr(selector_native.selector, "load_wrong_selector_functions", lambda path: {"inject_wrong_service_selector": inject})
    monkeypatch.setattr(selector_native, "application_checks", lambda *args: actual in ("healthy", "known_repair"))
    arguments = (Command(), ["kubectl"], tmp_path, manifests, tmp_path, before, {"pinned": {digest}},
                 lambda *args: {"after": True}, selector_transfer.CONDITIONS.index(actual) + 1, owner)
    if condition == "ineffective_fault":
        with pytest.raises(ValueError, match="did not break the actual workflow"):
            selector_native.run_cycle(*arguments)
        assert "/actor" not in rpc_paths and "/check" not in rpc_paths
        assert len(patches) == 1
    else:
        result = selector_native.run_cycle(*arguments)
        assert result["control_passed"] and result["grade"]["success"] is (actual in ("healthy", "known_repair"))
        assert len(patches) == {"healthy": 0, "known_repair": 2, "restart_only": 2, "premature_finish": 1}[actual]
        assert (tmp_path / "independent-journal-review.json").exists()
    assert json.loads((tmp_path / "port-forward-cleanup.json").read_text())["exited"]
    assert (tmp_path / "source.jsonl").exists() and (tmp_path / "relay.jsonl").exists()
    assert any("can-i" in call and "pods" in call and "--subresource=exec" in call for call in calls)


@pytest.mark.parametrize("damage", ["stateful_spec", "uid", "resource_set", "unrelated_restart", "frontend_restart"])
def test_native_application_guard_rejects_unrelated_changes(monkeypatch, damage):
    deployment = {"metadata": {"name": "valkey-cart", "uid": "store"}, "spec": {"replicas": 1}}
    pod = {"metadata": {"uid": "store-pod", "labels": {"opentelemetry.io/name": "valkey-cart"}},
           "status": {"containerStatuses": [{"name": "store", "containerID": "stable", "restartCount": 0}]}}
    before = {"deployments": [deployment], "daemonsets": [], "services": [], "pods": [pod]}
    if damage == "frontend_restart":
        before["pods"][0]["metadata"]["labels"]["opentelemetry.io/name"] = "frontend"
    after = copy.deepcopy(before)
    if damage == "stateful_spec":
        after["deployments"][0]["spec"]["replicas"] = 2
    elif damage == "uid":
        after["deployments"][0]["metadata"]["uid"] = "replacement"
    elif damage == "resource_set":
        after["deployments"] = []
    else:
        after["pods"][0]["status"]["containerStatuses"][0]["restartCount"] = 1
    monkeypatch.setattr(astronomy_host, "controller_ready", lambda obj: True)
    monkeypatch.setattr(astronomy_host, "validate_pod", lambda *args: None)
    with pytest.raises(ValueError, match="changed|restarted"):
        selector_native.application_checks(before, after, {})
