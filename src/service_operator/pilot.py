from pathlib import Path

from service_operator import hotel, tools


TASK = "Investigate this hotel workflow alert. Keep existing bookings intact, verify search and booking, and decide whether changes are needed."
CASES = ("healthy", "selector", "retry")


def public_task():
    return {"task": TASK, "tools": tools.tool_definitions("pilot"),
            "action_limit_s": 300, "max_tool_calls": 64, "max_checks": 2,
            "completion": "declares completion; workload and private evaluation continue",
            "policy_updates": "request a rollout; inspect generation/readiness and runtime metrics before assuming application"}


def grade(case, public, evaluation, recovered, replayed, topology_ok, safe_policy, post_completion):
    if case not in CASES:
        raise ValueError("unknown private case")
    if case != "retry" and replayed is not None:
        raise ValueError("replay is only defined for retry cases")
    on_time = (public.finished_at is not None
               and 0 <= public.finished_at - public.started < public.deadline_s)
    no_change = case != "healthy" or public.mutation_attempts == 0
    checks = {"declared_on_time": on_time, "recovered": recovered,
              "replay_satisfied": case != "retry" or replayed is True,
              "topology_ok": topology_ok, "safe_policy": safe_policy,
              "invariants_preserved": evaluation["invariants_preserved"],
              "baseline_workflow_valid": evaluation["baseline_valid"],
              "final_workflow_valid": evaluation["final_valid"],
              "healthy_no_mutations": no_change, "post_completion_work": post_completion >= 2}
    return {"success": all(value is True for value in checks.values()), **checks,
            "replay_required": case == "retry", "replay_passed": replayed,
            "post_completion_searches": post_completion, "tool_calls": public.calls,
            "diagnostic_checks": public.checks, "mutation_attempts": public.mutation_attempts,
            "successful_mutations": public.mutations}


def client_objects(namespace, app_namespace, image):
    client = hotel.deployment("client", "client", image, "128Mi",
                              ["python", "/opt/operator/relay.py", f"http://frontend.{app_namespace}.svc.cluster.local:5000"])
    client["metadata"]["namespace"] = namespace
    spec = client["spec"]["template"]["spec"]
    spec["volumes"] = [{"name": "relay-code", "configMap": {"name": "relay-code"}}]
    spec["containers"][0]["volumeMounts"] = [{"name": "relay-code", "mountPath": "/opt/operator", "readOnly": True}]
    spec["containers"][0]["readinessProbe"] = {"tcpSocket": {"port": 8080}, "periodSeconds": 1, "timeoutSeconds": 1}
    code = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "relay-code", "namespace": namespace},
            "data": {"relay.py": Path(__file__).with_name("relay.py").read_text()}}
    service = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "client", "namespace": namespace},
               "spec": {"selector": {"io.kompose.service": "client"}, "ports": [{"port": 8080, "targetPort": 8080}]}}
    return [code, client, service]
