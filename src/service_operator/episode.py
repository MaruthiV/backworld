import json
import subprocess
import time
from pathlib import Path

from service_operator import hotel, http_probe, workflow
from service_operator.kubernetes import save


def routing_state(runtime, namespace):
    state = {}
    for key, resource in (("service", "service/frontend"), ("deployments", "deployments"),
                          ("pods", "pods"), ("replicasets", "replicasets"), ("slices", "endpointslices")):
        args = ["-n", namespace, "get", resource, "-o", "json"]
        if key == "slices":
            args += ["-l", "kubernetes.io/service-name=frontend"]
        state[key] = json.loads(runtime.kube(args))
    return state


class BookingLedger:
    def __init__(self, runtime, namespace, private_namespace, output):
        self.runtime, self.namespace, self.private_namespace = runtime, namespace, private_namespace
        self.output = output
        self.output.mkdir()
        self.remote_code = Path(http_probe.__file__).read_text()
        self.events = []
        self.before = self.snapshot()
        save(output / "initial.json", self.before)

    def http(self, url):
        raw = self.runtime.kube(["-n", self.private_namespace, "exec", "deployment/client", "--", "python", "-c",
                                 self.remote_code, url, "1"], timeout=15)
        return json.loads(raw)

    def rpc(self, action):
        code = ("import urllib.request;"
                f"r=urllib.request.Request('http://127.0.0.1:8099/{action}',data=b'',method='POST');"
                "print(urllib.request.urlopen(r,timeout=5).read().decode())")
        raw = self.runtime.kube(["-n", self.private_namespace, "exec", "deployment/client", "--", "python", "-c", code], timeout=15)
        return json.loads(raw)

    def snapshot(self):
        return json.loads(self.runtime.kube(["-n", self.namespace, "exec", "deployment/mongodb-reservation", "--",
                                            "mongo", "--quiet", "--eval", hotel.SNAPSHOT_JS]))

    def evaluate(self, events, name):
        immediate = self.snapshot()
        save(self.output / f"{name}-immediate.json", immediate)
        time.sleep(0.25)
        after = self.snapshot()
        save(self.output / f"{name}-settled.json", after)
        for event, verdict in zip(events, workflow.check_window(self.before, immediate, after, events)):
            event["verdict"] = verdict
        save(self.output / f"{name}-requests.json", events)
        return events

    def probe(self, origin):
        if origin != "baseline":
            raise ValueError("only baseline checks use controller snapshots")
        return self.evaluate([self.rpc("baseline")], "baseline")[0]

    def public_check(self):
        event = self.rpc("check")
        save(self.output / "diagnostic.json", event)
        return {"http_status": event["response"]["status"], "body": event["response"]["body"],
                "elapsed_s": event["response"]["elapsed_s"]}

    def complete(self):
        self.rpc("finish")


class PrivateWorkload:
    def __init__(self, ledger):
        self.ledger = ledger
        self.count = 12
        self.events = []
        self.error = None
        self.started = None
        self.driver_state = None

    def start(self):
        self.started = time.monotonic()
        self.ledger.rpc("start")

    def join(self):
        time.sleep(max(0, self.started + self.count * 3 - time.monotonic()))
        deadline = time.monotonic() + 60
        failures = []
        while time.monotonic() < deadline:
            try:
                state = self.ledger.rpc("status")
            except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as exc:
                failures.append({"error": f"{type(exc).__name__}: {exc}", "unix_s": time.time()})
                save(self.ledger.output / "collection-errors.json", failures)
                time.sleep(2)
                continue
            save(self.ledger.output / "driver.json", state)
            if state["ended_monotonic_s"] is not None:
                self.driver_state = state
                self.error = state["error"]
                self.ledger.events = self.ledger.evaluate(state["events"], "final")
                self.events = [event for event in self.ledger.events if event["origin"] == "scheduled"]
                return
            time.sleep(2)
        raise TimeoutError("private workload did not finish")


def grade_episode(workload, ledger, tools, route_ok, healthy):
    events = workload.events
    complete = workload.error is None and len(events) == workload.count
    invariant = bool(ledger.events) and all(event.get("verdict", {}).get("invariant_ok", False) for event in ledger.events)
    tail_ok = len(events) >= 3 and all(event["verdict"]["accepted"] for event in events[-3:])
    declared = workload.driver_state["declared_complete_monotonic_s"]
    post_done = sum(event["offered_monotonic_s"] > declared for event in events) if declared is not None else 0
    healthy_ok = not healthy or (tools.mutations == 0 and all(event["verdict"]["accepted"] for event in events))
    return {"success": bool(complete and invariant and route_ok and tail_ok and healthy_ok
                            and tools.finished_at is not None and declared is not None and post_done >= 2),
            "workload_complete": complete, "invariants_preserved": invariant, "route_ok": route_ok,
            "tail_ok": tail_ok, "post_completion_requests": post_done,
            "scheduled_valid": sum(event["verdict"]["accepted"] for event in events),
            "scheduled_attempts": len(events), "configuration_mutations": tools.mutations,
            "declared_complete": tools.finished_at is not None}
