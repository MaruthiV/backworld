import time

import pytest

from service_operator.tools import PublicTools


class Backend:
    def __init__(self):
        self.calls = []

    def perform(self, name, args):
        self.calls.append((name, args))
        return {"acknowledged": True}


def session(backend=None):
    backend = backend or Backend()
    events = []
    tools = PublicTools(backend, lambda: {"http_status": 200}, events.append, time.monotonic())
    return tools, backend, events


@pytest.mark.parametrize("call", [
    {"tool": "inspect", "arguments": {"resource": "pods", "name": "frontend", "namespace": "judge"}},
    {"tool": "inspect", "arguments": {"resource": "secrets", "name": "frontend"}},
    {"tool": "inspect", "arguments": {"resource": "pods", "name": "client"}},
    {"tool": "logs", "arguments": {"deployment": "../../judge", "lines": 10}},
    {"tool": "logs", "arguments": {"deployment": "frontend", "lines": True}},
    {"tool": "restart", "arguments": {"deployment": "frontend", "privileged": True}},
    {"tool": "restart", "arguments": {"deployment": "frontend", "patch": {"hostNetwork": True}}},
    {"tool": "set_selector", "arguments": {"service": "frontend", "selector": {}}},
    {"tool": "set_selector", "arguments": {"service": "frontend", "selector": {"app": "x;cat /etc/passwd"}}},
    {"tool": "set_selector", "arguments": {"service": "frontend", "selector": {"a..b/key": "x"}}},
    {"tool": "kubectl", "arguments": {"command": "get secrets -A"}},
    {"tool": "read_file", "arguments": {"path": "docs/HANDOFF.md"}},
    {"tool": "grade", "arguments": {}},
    {"tool": "wait", "arguments": {"seconds": 60}},
])
def test_forbidden_call_never_reaches_backend(call):
    tools, backend, events = session()
    assert not tools.invoke(call)["ok"]
    assert backend.calls == []
    assert len(events) == 1


def test_selector_supports_equivalent_choices_without_returning_private_data():
    tools, backend, events = session()
    call = {"tool": "set_selector", "arguments": {"service": "frontend", "selector": {"app": "alternate"}}}
    assert tools.invoke(call)["ok"]
    assert backend.calls == [("set_selector", call["arguments"])]
    assert tools.mutations == 1


def test_backend_failure_does_not_expose_its_message():
    class BrokenBackend:
        def perform(self, name, args):
            raise ValueError("private evaluator path /judge/expected.json")

    tools, _, _ = session(BrokenBackend())
    result = tools.invoke({"tool": "inspect", "arguments": {"resource": "service", "name": "frontend"}})
    assert result["error"]["code"] == "operation_failed"
    assert "judge" not in str(result)


def test_completion_closes_actions_without_granting_success():
    tools, backend, events = session()
    done = tools.invoke({"tool": "finish", "arguments": {"summary": "fixed"}})
    assert done == {"ok": True, "result": {"declared_complete": True}}
    assert not tools.invoke({"tool": "restart", "arguments": {"deployment": "frontend"}})["ok"]
    assert backend.calls == []


def test_calls_and_functional_checks_are_bounded():
    tools, backend, events = session()
    for _ in range(2):
        assert tools.invoke({"tool": "check", "arguments": {}})["ok"]
    assert not tools.invoke({"tool": "check", "arguments": {}})["ok"]
    tools.max_calls = tools.calls
    assert not tools.invoke({"tool": "inspect", "arguments": {"resource": "service", "name": "frontend"}})["ok"]
    assert backend.calls == []
