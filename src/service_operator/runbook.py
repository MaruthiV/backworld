def investigate(send):
    def call(tool, **args):
        reply = send({"tool": tool, "arguments": args})
        if not reply["ok"]:
            raise RuntimeError("public operation failed: " + tool)
        return reply["result"]

    def endpoints():
        result = call("inspect", resource="endpoints", name="frontend")
        return {endpoint.get("targetRef", {}).get("name")
                for part in result["slices"] for endpoint in part["endpoints"]
                if endpoint.get("conditions", {}).get("ready") is True}

    def metrics():
        return {**call("metrics", service="search")["values"],
                **call("metrics", service="rate")["values"]}

    def amplification(before, after):
        count = after["search_requests_total"] - before["search_requests_total"]
        attempts = after["search_rate_attempts_total"] - before["search_rate_attempts_total"]
        if count <= 0 or attempts < 0:
            raise RuntimeError("insufficient stable traffic observations")
        return attempts / count

    call("clock")
    service = call("inspect", resource="service", name="frontend")
    frontend = call("inspect", resource="deployment", name="frontend")
    search = call("inspect", resource="deployment", name="search")
    pods = call("inspect", resource="pods", name="frontend")["pods"]
    ready = {pod["name"] for pod in pods if any(
        row.get("type") == "Ready" and row.get("status") == "True" for row in pod["conditions"])}
    if not ready:
        raise RuntimeError("no ready frontend pods")
    if endpoints() != ready:
        unique = {key: value for key, value in frontend["pod_labels"].items()
                  if search["pod_labels"].get(key) != value}
        if not unique or service["selector"] == unique:
            raise RuntimeError("no justified selector correction")
        call("set_selector", service="frontend", selector=unique)
        for _ in range(6):
            if endpoints() == ready:
                break
            call("wait", seconds=2)
        else:
            raise RuntimeError("frontend routing did not converge")

    before = metrics()
    call("wait", seconds=5)
    after = metrics()
    ratio = amplification(before, after)
    if min(before["rate_queue_depth"], after["rate_queue_depth"]) >= 20 and ratio >= 2:
        call("configure_rate", qps_limit=50, queue_capacity=256)
        call("configure_search", timeout_ms=1000, max_attempts=2,
             backoff_ms=50, backoff_multiplier=2.0, jitter=0.2)
        for _ in range(12):
            deployments = [call("inspect", resource="deployment", name=name) for name in ("rate", "search")]
            if all(row["status"].get("readyReplicas") == row["status"].get("updatedReplicas")
                   == row["status"].get("replicas") == 1
                   and row["status"].get("observedGeneration", 0) >= row["generation"] for row in deployments):
                break
            call("wait", seconds=2)
        else:
            raise RuntimeError("policy rollout did not converge")
        call("wait", seconds=10)
        call("wait", seconds=10)
        before = metrics()
        call("wait", seconds=5)
        after = metrics()
        ratio = amplification(before, after)
    if after["rate_queue_depth"] > 5 or ratio > 2.1:
        raise RuntimeError("service has not recovered")
    receipt = call("check")
    if (not receipt["search"] or receipt["search"].get("status") != 200
            or not receipt["search"].get("body", {}).get("features")
            or not receipt["booking"] or receipt["booking"].get("status") != 200):
        raise RuntimeError("workflow check did not succeed")
    call("finish", summary="routing, live traffic and booking checked")
