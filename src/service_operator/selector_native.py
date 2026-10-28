import base64
import copy
import hashlib
import http.client
import json
import math
import re
import subprocess
import time
from types import SimpleNamespace

from service_operator import astronomy_host, cart, selector, selector_transfer, shop_selector, tools


def audit(events, relay_events, run_id):
    def indexed(stage):
        rows = [e for e in events if e["stage"] == stage]
        result = {r["request_id"]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate private journal identity: " + stage)
        return result

    intents, responses = indexed("intent"), indexed("response")
    if not intents or set(intents) != set(responses):
        raise ValueError("source HTTP intent/response accounting differs")
    admissions, completions = {}, {}
    for event in relay_events:
        target = admissions if event["stage"] == "started" else completions if event["stage"] == "completed" else None
        identity = event["client_request_id"]
        if target is None or identity in target or identity not in intents:
            raise ValueError("unknown or duplicate relay admission")
        target[identity] = event
    if set(admissions) != set(completions):
        raise ValueError("relay admission has no terminal receipt")
    delivered = {}
    for identity, event in responses.items():
        response, intent = event["response"], intents[identity]
        if response["request_id"] != identity or not identity.startswith(run_id + "-"):
            raise ValueError("source identity changed")
        if "raw_body_b64" in response:
            raw = base64.b64decode(response["raw_body_b64"], validate=True)
            try:
                decoded = json.loads(raw)
            except (ValueError, UnicodeError):
                if not response.get("error"):
                    raise ValueError("unreported invalid JSON response")
            else:
                if decoded != response["body"]:
                    raise ValueError("source body differs from raw HTTP bytes")
        admission, completed = admissions.get(identity), completions.get(identity)
        delivered[identity] = False
        if response.get("admission_id") is not None:
            if (not admission or not completed or admission["id"] != response["admission_id"]
                    or admission["id"] != completed["id"] or completed["status"] != response["status"]
                    or admission["request_hash"] != hashlib.sha256(intent["path"].encode()).hexdigest()
                    or admission["method"] != intent["method"]
                    or admission["request_body_sha256"] != hashlib.sha256(b"" if intent["body"] is None else json.dumps(intent["body"]).encode()).hexdigest()):
                raise ValueError("source and relay receipts disagree")
            if "raw_body_b64" in response and completed["body_sha256"] != hashlib.sha256(raw).hexdigest():
                raise ValueError("source and relay response bytes disagree")
            delivered[identity] = completed["delivered"] is True and not bool(response.get("error"))
        elif not response.get("error"):
            raise ValueError("received response has no relay admission")

    summaries = [e["summary"] for e in events if e["stage"] == "summary"]
    starts = [e for e in events if e["stage"] == "source_start"]
    actors = [e for e in events if e["stage"] == "actor_start"]
    finishes = [e for e in events if e["stage"] == "finish"]
    if len(summaries) != 1 or "error" in summaries[0] or len(starts) != 1 or len(actors) != 1 or len(finishes) > 1:
        raise ValueError("source lifecycle is incomplete")
    summary, opened, actor = summaries[0], starts[0]["opened_s"], actors[0]["actor_opened_s"]
    offer_end, ended = actors[0]["offer_end_s"], actors[0]["ended_s"]
    if (not all(type(x) in (int, float) and math.isfinite(x) for x in (opened, actor, offer_end, ended))
            or not 0 <= actor - opened <= 60 or offer_end != actor + 330 or ended != offer_end + 5
            or starts[0]["interval_s"] != 0.5 or starts[0].get("traffic_protocol_version") != 2 or starts[0].get("drain_s") != 5):
        raise ValueError("source clocks or fixed horizon differ")
    offers, outcomes = indexed("offer"), indexed("outcome")
    count = math.ceil((offer_end - opened) / 0.5)
    if set(offers) != {f"slot-{i}" for i in range(count)} or set(outcomes) != set(offers):
        raise ValueError("scheduled work was omitted or duplicated")
    prior = "bw-cart-" + run_id + "-prior"
    additions = [e for e in events if e["stage"] == "addition"]
    seed = next((e for e in additions if e["user"] == prior), None)
    if seed is None or seed["accepted"] is not True:
        raise ValueError("acknowledged seed state is missing")
    expected = seed["expected"]
    if expected["userId"] != prior or len(expected["items"]) != 1 or expected["items"][0]["quantity"] != 2:
        raise ValueError("seed does not match the protected workflow")
    product = expected["items"][0]["productId"]
    protected = {i for i,v in intents.items() if v["phase"] == "protected"}
    if {v["http_request_id"] for v in outcomes.values()} != protected or len(protected) != count:
        raise ValueError("scheduled slots do not match actual source HTTP attempts")
    checked = []
    for index in range(count):
        offer, outcome = offers[f"slot-{index}"], outcomes[f"slot-{index}"]
        if abs(offer["offered_s"] - (opened + index * 0.5)) > 1e-6 or offer["offered_s"] >= offer_end or offer["deadline_s"] != 5:
            raise ValueError("source schedule changed")
        identity = outcome["http_request_id"]
        response, intent = responses[identity]["response"], intents[identity]
        if intent["method"] != "GET" or intent["body"] is not None:
            raise ValueError("protected traffic must be read only")
        path = "/api/products?currencyCode=USD" if index % 2 == 0 else "/api/cart?" + shop_selector.urlencode({"sessionId": prior, "currencyCode": "USD"})
        if intent["path"] != path or intent["started_s"] < offer["offered_s"] or outcome["finished_s"] < responses[identity]["finished_s"]:
            raise ValueError("protected request path or source clock differs")
        valid = shop_selector.catalog_valid(response, product) if index % 2 == 0 else cart.matches(response, expected, enriched=True)
        if outcome["valid"] is not valid or outcome["delivered"] is not delivered[identity]:
            raise ValueError("protected verdict differs from raw HTTP/relay data")
        checked.append({**outcome, "valid": valid, "delivered": delivered[identity]})
    loss = selector_transfer.workload_loss(list(offers.values()), checked, opened, ended)
    if loss != summary["loss"]:
        raise ValueError("source loss rollup differs")

    users = set()
    for addition in additions:
        user, expected_cart = addition["user"], addition["expected"]
        if user in users or not user.startswith("bw-cart-" + run_id + "-"):
            raise ValueError("write identity was reused")
        users.add(user)
        quantity = 2 if user == prior else 1
        expected_cart_fixed = {"userId": user, "items": [{"productId": product, "quantity": quantity}]}
        wire = astronomy_host.expected_wire(user, product, quantity)
        after = addition["snapshot"]
        if (addition["accepted"] is not True or expected_cart != expected_cart_fixed or after["raw_hex"] != wire
                or after["cart"] != expected_cart or after["key_type"] != "hash" or after["hash_fields"] != 1
                or not 0 < after["ttl_ms"] <= cart.MAX_TTL_MS):
            raise ValueError("acknowledged cart differs from independent wire/state check")
        attempts = [e for e in intents.values() if e["method"] == "POST" and e["body"].get("userId") == user]
        if len(attempts) != 1 or attempts[0]["path"] != "/api/cart" or attempts[0]["body"] != {"userId": user, "item": expected_cart["items"][0]}:
            raise ValueError("cart write was retried or changed")
        ids = addition["response_ids"]
        if len(ids) != 3 or not all(delivered[i] for i in ids) or not shop_selector.catalog_valid(responses[ids[0]]["response"], product):
            raise ValueError("addition HTTP responses were not received")
        if (not cart.matches(responses[ids[1]]["response"], expected_cart)
                or not cart.matches(responses[ids[2]]["response"], expected_cart, enriched=True)):
            raise ValueError("addition acknowledgement/readback is incorrect")
        samples = [e for e in events if e["stage"] == "store" and e["snapshot"]["user"] == user]
        baseline = next((i for i,e in enumerate(samples) if e["snapshot"] == after), None)
        if baseline is None or baseline == 0 or samples[0]["snapshot"]["key_type"] != "none" or len(samples[baseline:]) < 2:
            raise ValueError("cart freshness or continued state observations are missing")
        first = samples[baseline]
        for sample in samples[baseline:]:
            if not cart.preserved(first["snapshot"], sample["snapshot"], sample["observed_s"] - first["observed_s"]):
                raise ValueError("acknowledged cart bytes or expiry changed")
    if {e["body"]["userId"] for e in intents.values() if e["method"] == "POST"} != users:
        raise ValueError("unaccounted or uncertain write")
    tail = [o for o in offers.values() if o["offered_s"] >= offer_end - 10]
    recovered = len(tail) >= 20 and all(outcomes[o["request_id"]]["valid"] and outcomes[o["request_id"]]["delivered"]
        and outcomes[o["request_id"]]["finished_s"] <= min(ended, o["offered_s"] + 5) for o in tail)
    finished = finishes[0]["finished_s"] if finishes else None
    post = sum(o["offered_s"] >= finished for o in offers.values()) if finished is not None else 0
    workflow = "bw-cart-" + run_id + "-final" in users
    independent = {"loss": loss, "workflow_valid": workflow, "state_preserved": True,
                   "recovered": recovered, "post_completion": post,
                   "source_on_time": finished is not None and 0 <= finished - actor < 300,
                   "opened_s": opened, "actor_opened_s": actor, "finished_s": finished, "offer_end_s": offer_end, "ended_s": ended}
    if independent != summary:
        raise ValueError("private summary differs from independent journal review")
    probes = [identity for identity, intent in intents.items() if intent["phase"] == "effect_probe"
              and intent["started_s"] >= opened and responses[identity]["finished_s"] <= actor]
    effect_confirmed = len(probes) >= 2 and all(delivered[i] and not shop_selector.catalog_valid(responses[i]["response"], product)
                                               for i in probes[-2:])
    return {**independent, "http_attempts": len(intents), "relay_admissions": len(admissions),
            "unadmitted_failed_attempts": len(intents) - len(admissions), "exact_writes": len(users),
            "fault_effect_confirmed": effect_confirmed,
            "max_dispatch_lag_s": max(intents[o["http_request_id"]]["started_s"] - offers[o["request_id"]]["offered_s"] for o in checked)}


def application_checks(before, after, images, allow_frontend_restart=False):
    if not all(astronomy_host.controller_ready(o) for o in after["deployments"] + after["daemonsets"]):
        raise ValueError("application controller is not current and Ready")
    for pod in after["pods"]:
        astronomy_host.validate_pod(pod, images)
    for resource in ("services", "deployments", "daemonsets"):
        old = {o["metadata"]["name"]: o for o in before[resource]}
        new = {o["metadata"]["name"]: o for o in after[resource]}
        if set(old) != set(new):
            raise ValueError("application resources changed")
        for name, prior in old.items():
            actual = new[name]
            if prior["metadata"]["uid"] != actual["metadata"]["uid"]:
                raise ValueError("application resource UID changed")
            a, b = copy.deepcopy(prior["spec"]), copy.deepcopy(actual["spec"])
            if name == "frontend" and resource == "services":
                a.pop("selector", None)
                b.pop("selector", None)
            if name == "frontend" and resource == "deployments":
                for spec in (a, b):
                    annotations = spec["template"]["metadata"].get("annotations", {})
                    annotations.pop("kubectl.kubernetes.io/restartedAt", None)
                    if not annotations:
                        spec["template"]["metadata"].pop("annotations", None)
            if a != b:
                raise ValueError("application spec changed outside the allowed selector/restart fields")
    def stable(data):
        return {p["metadata"]["uid"]: [(c["name"], c["containerID"], c["restartCount"]) for c in p["status"]["containerStatuses"]]
                for p in data["pods"] if not (allow_frontend_restart and p["metadata"].get("labels", {}).get("opentelemetry.io/name") == "frontend")}
    if stable(before) != stable(after):
        raise ValueError("an unrelated application Pod or container restarted")
    route_valid = True
    for name in ("frontend-proxy", "frontend", "cart", "product-catalog", "valkey-cart"):
        services = [s for s in after["services"] if s["metadata"]["name"] == name]
        if len(services) != 1:
            raise ValueError("workflow Service is missing or duplicated")
        try:
            astronomy_host.validate_routing(services[0], after["endpointslices"], after["pods"], after["replicasets"], after["deployments"])
        except ValueError:
            if name != "frontend":
                raise
            route_valid = False
    return route_valid


def run_cycle(command, kube, bundle, manifest, folder, before, images, inventory, cycle, owner):
    condition = selector_transfer.CONDITIONS[cycle - 1]
    namespace, private = manifest["application_namespace"], manifest["private_namespace"]
    process = pod = None
    rpc_count = 0

    def save(name, value):
        (folder / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

    def rpc(path):
        nonlocal rpc_count
        rpc_count += 1
        receipt = {"path": path, "host_started_s": time.monotonic()}
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=75 if path == "/seed" else 20)
        try:
            connection.request("POST", path, body=b"")
            response = connection.getresponse()
            receipt.update(status=response.status, value=json.loads(response.read(1048576)))
            if response.status != 200:
                raise RuntimeError("private RPC failed; no automatic retry")
            return receipt["value"]
        finally:
            connection.close()
            receipt["host_finished_s"] = time.monotonic()
            save(f"rpc-{rpc_count:04d}.json", receipt)

    try:
        job_plan = next(o for o in owner.objects if o["kind"] == "Job")
        job = owner.get(job_plan)
        job_uid = astronomy_host.owned(job, job_plan, manifest["run_id"], owner.receipts[astronomy_host.identity(job_plan)["key"]]["uid"])
        save("job.json", job)
        deadline = time.monotonic() + 90
        while True:
            pods = json.loads(command(kube + ["get", "pods", "-n", private, "-o", "json"]))["items"]
            if len(pods) == 1:
                pod = pods[0]
                if any(c.get("type") == "Ready" and c.get("status") == "True" for c in pod.get("status", {}).get("conditions", [])):
                    astronomy_host.validate_pod(pod, images)
                    if (pod["metadata"].get("labels", {}).get("backworld.run") != manifest["run_id"]
                            or not any(o.get("kind") == "Job" and o.get("uid") == job_uid and o.get("controller")
                                       for o in pod["metadata"].get("ownerReferences", []))
                            or {c["name"] for c in pod["spec"]["containers"]} != {"source", "relay"}
                            or pod["spec"].get("automountServiceAccountToken") is not False
                            or any(c["restartCount"] != 0 for c in pod["status"]["containerStatuses"])):
                        raise ValueError("private helper ownership, token boundary or source identity differs")
                    break
            if time.monotonic() >= min(deadline, command.deadline):
                raise TimeoutError("private selector helper did not become Ready")
            time.sleep(1)
        save("private-pod-before.json", pod)
        log_path = folder / "port-forward.log"
        args = kube + ["port-forward", "-n", private, "pod/" + pod["metadata"]["name"], ":8081", "--address=127.0.0.1"]
        save("port-forward.json", {"args": args})
        with log_path.open("w") as log:
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=command.env)
        deadline = time.monotonic() + 15
        while True:
            match = re.search(r"Forwarding from 127\.0\.0\.1:(\d+) -> 8081", log_path.read_text())
            if match:
                port = int(match[1])
                break
            if process.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("owned private tunnel failed")
            time.sleep(0.1)
        runtime = SimpleNamespace(kube=lambda args, **kwargs: command(kube + args, **kwargs))
        backend = selector_transfer.SelectorBackend(runtime, namespace, "shop")
        permissions = []
        for verb, resource, ns, expected in [("patch", "services/frontend", namespace, True),
                ("patch", "deployments.apps/frontend", namespace, True), ("patch", "services/valkey-cart", namespace, False),
                ("patch", "deployments.apps/valkey-cart", namespace, False), ("get", "secrets", namespace, False),
                ("create", "pods/exec", namespace, False), ("get", "pods", private, False)]:
            target, extra = ("pods", ["--subresource=exec"]) if resource == "pods/exec" else (resource, [])
            result = command(kube + backend.identity + ["auth", "can-i", verb, target, "-n", ns, *extra], check=False).strip()
            permissions.append({"verb": verb, "resource": resource, "namespace": ns, "expected": expected, "result": result})
        save("native-permissions.json", permissions)
        if any(row["result"] != ("yes" if row["expected"] else "no") for row in permissions):
            raise ValueError("native actor permissions differ")
        rpc("/seed")
        if rpc("/probe")["valid"] is not True:
            raise ValueError("healthy Service-path workflow did not pass")
        rpc("/start")
        effect = []
        if condition != "healthy":
            class Injector:
                def get_service_json(self, name, ns):
                    if (name, ns) != ("frontend", namespace):
                        raise ValueError("unexpected injection target")
                    return json.loads(command(kube + ["get", "service", name, "-n", ns, "-o", "json"]))

                def patch_service(self, name, ns, obj):
                    if (name, ns) != ("frontend", namespace):
                        raise ValueError("unexpected injection target")
                    original = next(o for o in before["services"] if o["metadata"]["name"] == name)
                    if obj["metadata"]["uid"] != original["metadata"]["uid"] or obj["spec"]["selector"] != {**original["spec"]["selector"], "current_service_name": name}:
                        raise ValueError("pinned injection effect differs")
                    patch = [{"op": "test", "path": "/metadata/uid", "value": obj["metadata"]["uid"]},
                             {"op": "test", "path": "/metadata/resourceVersion", "value": obj["metadata"]["resourceVersion"]},
                             {"op": "replace", "path": "/spec/selector", "value": obj["spec"]["selector"]}]
                    save("injection.json", {"source": "private-injector.py", "patch": patch})
                    command(kube + ["patch", "service", name, "-n", ns, "--type=json", "-p", json.dumps(patch)])
            functions = selector.load_wrong_selector_functions(bundle / "private-injector.py")
            functions["inject_wrong_service_selector"](SimpleNamespace(kubectl=Injector(), namespace=namespace), ["frontend"])
            deadline = time.monotonic() + 35
            while time.monotonic() < deadline:
                effect.append(rpc("/probe"))
                if len(effect) >= 2 and all(e["valid"] is False and e["response"].get("admission_id") for e in effect[-2:]):
                    break
                time.sleep(1)
            save("fault-effect.json", effect)
            if len(effect) < 2 or not all(e["valid"] is False and e["response"].get("admission_id") for e in effect[-2:]):
                raise ValueError("injection did not break the actual workflow; no proxy restart or actor run")
        host_opened = time.monotonic()
        rpc("/actor")
        def record_public(event):
            with (folder / "public-tools.jsonl").open("a") as handle:
                handle.write(json.dumps(event) + "\n")

        actor = tools.PublicTools(backend, lambda: rpc("/check"), record_public,
                host_opened, max_calls=64, complete=lambda: rpc("/finish"), profile="selector_shop", deadline_s=300)
        save("public-task.json", selector_transfer.public_task("shop"))
        actor_error = None
        try:
            if condition in ("healthy", "known_repair"):
                selector_transfer.investigate(actor.invoke)
            else:
                if condition == "restart_only":
                    if not actor.invoke({"tool": "restart", "arguments": {"deployment": "frontend"}})["ok"]:
                        raise RuntimeError("restart-only action failed")
                actor.invoke({"tool": "finish", "arguments": {"summary": "scripted completion control"}})
        except Exception as exc:
            actor_error = type(exc).__name__ + ": " + str(exc)
            save("actor-error.json", {"error": actor_error})
        deadline = time.monotonic() + 365
        while True:
            status = rpc("/status")
            if status["summary"] is not None:
                break
            if time.monotonic() >= min(deadline, command.deadline):
                raise TimeoutError("full source horizon did not complete")
            time.sleep(2)
        after = inventory(folder, "after")
        route = application_checks(before, after, images, condition == "restart_only")
        current = json.loads(command(kube + ["get", "pod", pod["metadata"]["name"], "-n", private, "-o", "json"]))
        save("private-pod-after.json", current)
        astronomy_host.validate_pod(current, images)
        if current["metadata"]["uid"] != pod["metadata"]["uid"] or current["status"]["containerStatuses"] != pod["status"]["containerStatuses"]:
            raise ValueError("private source/relay changed during the control")
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            save("port-forward-cleanup.json", {"pid": process.pid, "returncode": process.returncode, "exited": process.poll() is not None})
        if pod is not None:
            for name in ("source", "relay"):
                logs = command(kube + ["logs", pod["metadata"]["name"], "-n", private, "-c", name], check=False)
                (folder / (name + ".jsonl")).write_text(logs)
    events = [json.loads(line) for line in (folder / "source.jsonl").read_text().splitlines()]
    relay_events = [json.loads(line) for line in (folder / "relay.jsonl").read_text().splitlines()]
    reviewed = audit(events, relay_events, manifest["run_id"])
    save("independent-journal-review.json", reviewed)
    helper_manifests = [e for e in events if e["stage"] == "manifest"]
    if (len(helper_manifests) != 1 or helper_manifests[0]["source_sha256"] != manifest["source_hashes"]
            or helper_manifests[0]["run_id"] != manifest["run_id"] or helper_manifests[0]["namespace"] != namespace
            or helper_manifests[0].get("traffic_protocol_version") != 2 or helper_manifests[0].get("drain_s") != 5
            or any(e["stage"] == "rpc_error" for e in events)):
        raise ValueError("private source hashes differ or RPC failed")
    verdict = selector_transfer.grade(condition, actor, workflow_valid=reviewed["workflow_valid"],
        state_preserved=reviewed["state_preserved"], route_valid=route, application_healthy=True,
        recovered=reviewed["recovered"], post_completion=reviewed["post_completion"])
    verdict["success"] &= reviewed["source_on_time"]
    control_passed = (verdict["success"] is (condition in ("healthy", "known_repair"))
                      and actor_error is None and verdict["declared_on_time"]
                      and actor.mutations == actor.mutation_attempts == (1 if condition in ("known_repair", "restart_only") else 0)
                      and reviewed["state_preserved"] and reviewed["source_on_time"] and reviewed["post_completion"] >= 2
                      and (condition == "healthy" or reviewed["fault_effect_confirmed"])
                      and (condition != "healthy" or reviewed["loss"]["failed"] == 0)
                      and (condition in ("healthy", "known_repair") or not route and not reviewed["recovered"] and not reviewed["workflow_valid"]))
    result = {"condition": condition, "grade": verdict, "control_passed": control_passed, "journal": reviewed}
    save("selector-control.json", result)
    if not control_passed:
        raise ValueError("selector control did not match the required outcome")
    return result
