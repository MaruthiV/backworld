from pathlib import Path

from service_operator import hotel, tools


TASK = "Investigate this hotel workflow alert. Keep existing bookings intact, verify search and booking, and decide whether changes are needed."
CASES = ("healthy", "selector", "retry")


def public_task():
    return {"task": TASK, "tools": tools.tool_definitions("pilot"),
            "action_limit_s": 300, "max_tool_calls": 64, "max_checks": 2,
            "completion": "declares completion; workload and private evaluation continue",
            "policy_updates": "request a rollout; inspect generation/readiness and runtime metrics before assuming application"}


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
