import json
import math
import time
import uuid
from urllib.parse import urlencode

from service_operator import hotel, http_probe, retry, tools, workflow
from service_operator.kubernetes import save


METRICS = {
    "search": ("search_requests_total", "search_rate_attempts_total", "search_rate_timeouts_total",
               "search_rate_failures_total", "search_rate_successes_total"),
    "rate": ("rate_backend_qps_limit", "rate_queue_capacity", "rate_queue_depth", "rate_in_flight",
             "rate_requests_received_total", "rate_requests_completed_total", "rate_requests_expired_total",
             "rate_requests_rejected_total"),
}


def metric_services(namespace):
    return [{"apiVersion": "v1", "kind": "Service", "metadata": {"name": name + "-metrics", "namespace": namespace},
             "spec": {"selector": {"io.kompose.service": name},
                      "ports": [{"port": port, "targetPort": port}]}}
            for name, port in (("search", 9092), ("rate", 9091))]


class RetryBackend(tools.OperatorBackend):
    def perform(self, tool, args):
        if tool in tools.POLICY_FIELDS:
            name = "rate" if tool == "configure_rate" else "search"
            env = [{"name": spec[0], "value": str(args[key])} for key, spec in tools.POLICY_FIELDS[tool].items()]
            patch = {"spec": {"template": {"spec": {"containers": [{"name": name, "env": env}]}}}}
            self.kube(["patch", "deployment", name, "--type=strategic", "-p", json.dumps(patch)], timeout=15)
            return {"deployment": name, "requested_policy": args, "rollout_requested": True}
        if tool == "metrics":
            name = args["service"]
            raw = self.kube(["get", "--raw", f"/api/v1/namespaces/{self.namespace}/services/{name}-metrics/proxy/metrics"], timeout=15)
            values = {}
            for line in raw.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0] in METRICS[name]:
                    value = float(parts[1])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("invalid application metric")
                    values[parts[0]] = value
            if set(values) != set(METRICS[name]):
                raise ValueError("missing application metrics")
            return {"service": name, "values": values}
        if tool == "inspect" and args["resource"] == "deployment":
            name = args["name"]
            obj = json.loads(self.kube(["get", "deployment", name, "-o", "json"], timeout=15))
            result = {"name": name, "generation": obj["metadata"]["generation"], "replicas": obj["spec"].get("replicas"),
                      "status": obj.get("status", {}), "pod_labels": obj["spec"]["template"]["metadata"]["labels"]}
            policy = tools.POLICY_FIELDS.get("configure_" + name)
            if policy:
                container = next(c for c in obj["spec"]["template"]["spec"]["containers"] if c["name"] == name)
                env = {row["name"]: row.get("value") for row in container.get("env", [])}
                result["configured_policy"] = {key: env.get(spec[0]) for key, spec in policy.items()}
            return result
        return super().perform(tool, args)

    def kube(self, args, **kwargs):
        kwargs.setdefault("timeout", 15)
        return super().kube(args, **kwargs)


class SearchLedger:
    def __init__(self, runtime, namespace, workload, output):
        self.runtime, self.namespace, self.workload, self.output = runtime, namespace, workload, output
        output.mkdir()
        self.before = self.snapshot()
        self.events = []
        self.marker = uuid.uuid4().hex
        save(output / "initial.json", self.before)

    def snapshot(self):
        return json.loads(self.runtime.kube(["-n", self.namespace, "exec", "deployment/mongodb-reservation", "--",
                                            "mongo", "--quiet", "--eval", hotel.SNAPSHOT_JS], timeout=20))

    def probe(self, origin):
        event = {"id": len(self.events), "origin": origin, "started_s": time.monotonic(),
                 "expected": None, "response": None}
        self.events.append(event)
        path = self.output / f"probe-{event['id']:02d}.json"
        save(path, event)
        port = self.workload.frontend.local_port
        if port is None:
            raise RuntimeError("frontend transport is not ready")
        base = f"http://127.0.0.1:{port}"
        event["search"] = http_probe.request(base + self.workload._PATH, 4)
        save(path, event)
        body = event["search"].get("body")
        features = body.get("features") if isinstance(body, dict) and body.get("type") == "FeatureCollection" else None
        if (event["search"].get("status") == 200 and isinstance(features, list) and features
                and isinstance(features[0], dict) and isinstance(features[0].get("id"), str)):
            event["expected"] = {"hotelId": features[0]["id"], "customerName": self.marker + "-" + str(event["id"]),
                                 "inDate": "2015-04-09", "outDate": "2015-04-10", "number": 1}
            save(path, event)
            query = urlencode({**event["expected"], "username": "Cornell_1", "password": "1111111111"})
            event["response"] = http_probe.request(base + "/reservation?" + query, 4)
            save(path, event)
        event["finished_s"] = time.monotonic()
        save(path, event)
        return event

    def public_check(self):
        event = self.probe("diagnostic")
        return {"search": event["search"], "booking": event["response"]}

    def evaluate(self, catalog, name="final"):
        immediate = self.snapshot()
        save(self.output / (name + "-immediate.json"), immediate)
        time.sleep(0.25)
        after = self.snapshot()
        save(self.output / (name + "-settled.json"), after)
        old = {row["_id"]: row for row in self.before["reservations"]}
        new = {row["_id"]: row for row in after["reservations"]}
        added = [row for key, row in new.items() if key not in old]
        expected = {event["expected"]["customerName"] for event in self.events if event["expected"] is not None}
        invariant = (len(old) == len(self.before["reservations"]) and len(new) == len(after["reservations"])
                     and all(json.dumps(new.get(key), sort_keys=True) == json.dumps(row, sort_keys=True) for key, row in old.items())
                     and json.dumps(self.before["capacity"], sort_keys=True) == json.dumps(after["capacity"], sort_keys=True)
                     and json.dumps(immediate, sort_keys=True) == json.dumps(after, sort_keys=True)
                     and all(row.get("customerName") in expected for row in added))
        for event in self.events:
            try:
                retry.select_hotel(event["search"], catalog)
                search_ok = True
            except ValueError:
                search_ok = False
            if event["expected"] is None:
                event["verdict"] = {"accepted": False, "search_ok": search_ok, "booking_attempted": False}
                continue
            rows = [row for row in added if row.get("customerName") == event["expected"]["customerName"]]
            verdict = workflow.check_booking([], rows, event["expected"], event["response"])
            invariant &= (not rows or verdict["exact_new_write"]) and (not verdict["acknowledged"] or verdict["exact_new_write"])
            event["verdict"] = {**verdict, "search_ok": search_ok, "booking_attempted": True,
                                "accepted": verdict["accepted"] and search_ok}
        result = {"invariants_preserved": bool(invariant), "events": self.events,
                  "baseline_valid": self.events[0]["verdict"]["accepted"],
                  "final_valid": self.events[-1]["verdict"]["accepted"],
                  "confirmed_bookings": sum(e["verdict"]["accepted"] for e in self.events)}
        save(self.output / (name + "-evaluation.json"), result)
        return result


def grade(public, evaluation, recovered, replayed, topology_ok, safe_policy, healthy, post_completion):
    on_time = public.finished_at is not None and public.finished_at - public.started < public.deadline_s
    no_change = not healthy or public.mutation_attempts == 0
    success = (on_time and evaluation["invariants_preserved"] and evaluation["baseline_valid"] and evaluation["final_valid"]
               and recovered and (healthy or replayed is True) and topology_ok and safe_policy and no_change and post_completion >= 2)
    return {"success": bool(success), "declared_on_time": on_time, "recovered": recovered,
            "replay_required": not healthy, "replay_passed": replayed, "topology_ok": topology_ok, "safe_policy": safe_policy,
            "invariants_preserved": evaluation["invariants_preserved"], "final_workflow_valid": evaluation["final_valid"],
            "healthy_no_mutations": no_change, "post_completion_searches": post_completion,
            "tool_calls": public.calls, "diagnostic_checks": public.checks,
            "mutation_attempts": public.mutation_attempts, "successful_mutations": public.mutations}
