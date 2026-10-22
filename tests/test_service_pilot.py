import http.client
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from service_operator import pilot, relay, retry, retry_episode, tools


@contextmanager
def server(handler):
    instance = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance.server_address[1]
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


def request(port, path):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


@pytest.mark.parametrize("status,body", [(200, b'{"features":[]}'), (500, b'backend failed')])
def test_relay_preserves_upstream_failure_and_payload(status, body):
    paths, events = [], []

    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with server(Upstream) as source:
        with server(relay.handler(f"http://127.0.0.1:{source}", events.append)) as port:
            assert request(port, "/hotels?inDate=2015-04-09") == (status, body)
            assert request(port, "http://elsewhere/hotels")[0] == 404
            assert request(port, "/metrics")[0] == 404
    assert paths == ["/hotels?inDate=2015-04-09"]
    assert [e["stage"] for e in events] == ["started", "completed"]
    assert events[0]["id"] == events[1]["id"] and events[1]["delivered"]


def test_relay_does_not_retry_an_uncertain_booking():
    writes, events = [], []

    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self):
            writes.append(self.path)
            self.close_connection = True

        def log_message(self, *args):
            pass

    with server(Upstream) as source:
        with server(relay.handler(f"http://127.0.0.1:{source}", events.append)) as port:
            assert request(port, "/reservation?customerName=opaque")[0] == 502
    assert writes == ["/reservation?customerName=opaque"]
    assert events[-1]["status"] == 502 and events[-1]["error"] == "RemoteDisconnected"


def test_workload_id_is_visible_at_relay_admission(tmp_path):
    events = []
    recorder = retry.RecordedRequests(tmp_path / "http.jsonl", identify=True)

    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    with server(Upstream) as source:
        with server(relay.handler(f"http://127.0.0.1:{source}", events.append)) as port:
            response = recorder.get(f"http://127.0.0.1:{port}/hotels", timeout=2, headers={"X-Backworld-Request": "forged"})
    assert response.status_code == 200
    assert events[0]["client_request_id"] == recorder.snapshot()[0]["request_id"] == "workload-0"
    assert events[-1]["client_request_id"] == events[0]["client_request_id"]


def test_pilot_menu_and_bootstrap_have_no_case_parameter():
    contexts = [pilot.public_task() for _ in pilot.CASES]
    assert all(json.dumps(context, sort_keys=True) == json.dumps(contexts[0], sort_keys=True) for context in contexts)
    definitions = {row["name"]: row for row in contexts[0]["tools"]}
    assert {"set_selector", "configure_rate", "configure_search", "clock", "metrics"} <= set(definitions)
    assert all(name not in json.dumps(contexts[0]) for name in ("wrong_pod_selection", "retry_collapse", "healthy"))
    with pytest.raises(TypeError):
        pilot.public_task("retry")


@pytest.mark.parametrize("tool,args", [
    ("set_selector", {"service": "mongodb-reservation", "selector": {"app": "other"}}),
    ("set_selector", {"service": "frontend", "selector": {"app": "other"}, "namespace": "observer"}),
    ("restart", {"deployment": "mongodb-rate"}),
    ("configure_rate", {"qps_limit": True, "queue_capacity": 256}),
])
def test_pilot_combined_access_keeps_restricted_targets(tool, args):
    with pytest.raises(tools.ToolInputError):
        tools.validate_call({"tool": tool, "arguments": args}, "pilot")


def test_pilot_selector_uses_actor_identity_and_only_replaces_selector():
    commands = []
    backend = retry_episode.RetryBackend(SimpleNamespace(kube=lambda args, **kw: commands.append(args)), "app")
    name, args = tools.validate_call({"tool": "set_selector", "arguments": {
        "service": "frontend", "selector": {"io.kompose.service": "frontend"}}}, "pilot")
    backend.perform(name, args)
    assert "--as=system:serviceaccount:app:operator" in commands[0]
    assert json.loads(commands[0][-1]) == [{"op": "replace", "path": "/spec/selector", "value": {"io.kompose.service": "frontend"}}]
    rules = tools.operator_role("app", "pilot")[1]["rules"]
    writes = [rule for rule in rules if "patch" in rule["verbs"]]
    assert len(writes) == 2 and all(rule["resourceNames"] == ["frontend", "search", "rate"] for rule in writes)
