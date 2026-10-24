import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

from service_operator.astronomy_host import KubeControl, audit_helper, controller_ready, identity, memory_bytes, owned, validate_pod, validate_routing


TOOL_SHA256 = {
    "kind-linux-amd64": "aee6151561422756b764a4ae28e7f44cda5af5a9eead3cc9985112b1de8d8e0d",
    "kubectl-linux-amd64": "8efbb9435132a190920eb65a47a8c1ecf755ad85ab57a600c9bedbab460bb7a8",
}
NODE_IMAGE = "kindest/node:v1.34.11@sha256:44e222ee2132dab25ff87301682f89eb82c7880ea3a1bf543bfe9708fd08d67d"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(inputs, tools, output):
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((inputs / "package-summary.json").read_text())
    for name, expected in summary["files_sha256"].items():
        if sha(inputs / name) != expected:
            raise ValueError("prepared input hash changed: " + name)
    workspace = Path(__file__).resolve().parents[1]
    sources = ["src/service_operator/cart.py", "src/service_operator/astronomy_host.py",
               "experiments/service_astronomy_healthy.py", "experiments/service_astronomy_host.py"]
    for name in sources:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / name, target)
    for name, expected in TOOL_SHA256.items():
        if sha(tools / name) != expected:
            raise ValueError("Linux tool hash differs: " + name)
        shutil.copyfile(tools / name, output / name)
        (output / name).chmod(0o755)
    prior = json.loads((inputs / "ownership-and-cleanup-plan.json").read_text())
    run_id = uuid.uuid4().hex[:16]
    namespace, private = "bw-astro-" + run_id[:10], "bw-check-" + run_id[:10]
    replacements = {prior["run_id"]: run_id, prior["application_namespace"]: namespace, prior["private_namespace"]: private}

    def rewrite(value):
        if isinstance(value, str):
            return replacements.get(value, value)
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        if isinstance(value, dict):
            return {k: rewrite(v) for k,v in value.items()}
        return value

    app = rewrite(json.loads((inputs / "application-pinned.json").read_text()))
    control = rewrite(json.loads((inputs / "private-control.json").read_text()))
    config = next(o for o in control["items"] if o["kind"] == "ConfigMap")
    config["data"] = {"cart.py": (workspace / sources[0]).read_text(), "probe.py": (workspace / sources[2]).read_text()}
    job = next(o for o in control["items"] if o["kind"] == "Job")
    job["spec"]["activeDeadlineSeconds"] = 300
    pod = job["spec"]["template"]["spec"]
    init = copy.deepcopy(pod["containers"][0])
    init["name"] = "warmup"
    init["args"] = ["--namespace", namespace, "--run-id", run_id, "--output", "/evidence/warmup", "--readiness-only"]
    pod["initContainers"] = [init]
    save(output / "application.json", app)
    save(output / "control.json", control)
    shutil.copyfile(inputs / "image-lock.json", output / "image-lock.json")
    files = sources + list(TOOL_SHA256) + ["application.json", "control.json", "image-lock.json"]
    manifest = {"version": 1, "status": "prepared; not executed", "run_id": run_id,
                "application_namespace": namespace, "private_namespace": private, "cycles": 2,
                "files_sha256": {name: sha(output / name) for name in files},
                "source_hashes": {"cart.py": sha(workspace / sources[0]), "probe.py": sha(workspace / sources[2])},
                "limits": {"work_s": 3600, "cleanup_s": 300, "startup_s_per_cycle": 900,
                           "node_memory_bytes": 24 * 1024 ** 3, "node_cpus": 6, "node_pids": 8192,
                           "warmup_reads": 12, "warmup_wall_s": 75, "control_wall_s": 120, "job_deadline_s": 300},
                "required_host": {"linux_amd64": True, "cpu_count": 8, "memory_bytes": 30 * 1024 ** 3,
                                  "free_disk_bytes": 60 * 1024 ** 3, "dedicated_local_docker": True},
                "cloud_provisioning": "external and separately authorized; no account or provider operation in this runner"}
    save(output / "bundle.json", manifest)
    return manifest


class Commands:
    def __init__(self, output, env):
        self.output, self.env = output, env
        self.deadline = time.monotonic() + 3600
        self.count = 0

    def __call__(self, args, *, input=None, check=True):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("host command budget exhausted")
        self.count += 1
        prefix = self.output / f"command-{self.count:04d}"
        started = time.monotonic()
        receipt = {"args": args, "started_unix_s": time.time(), "input_sha256": hashlib.sha256(input.encode()).hexdigest() if input else None}
        try:
            result = subprocess.run(args, input=input, capture_output=True, text=True,
                                    timeout=min(20, remaining), env=self.env)
            prefix.with_suffix(".stdout").write_text(result.stdout)
            prefix.with_suffix(".stderr").write_text(result.stderr)
            receipt["returncode"] = result.returncode
        except subprocess.TimeoutExpired as exc:
            prefix.with_suffix(".stdout").write_bytes(exc.stdout or b"")
            prefix.with_suffix(".stderr").write_bytes(exc.stderr or b"")
            receipt["error"] = "timeout; mutation is not automatically retried"
            raise
        finally:
            receipt["elapsed_s"] = time.monotonic() - started
            save(prefix.with_suffix(".json"), receipt)
        if check and result.returncode:
            raise RuntimeError("command failed; see " + str(prefix))
        return result.stdout


def run(bundle, output, docker_context, host_id):
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise ValueError("execution requires a dedicated Linux AMD64 host")
    manifest = json.loads((bundle / "bundle.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if sha(bundle / name) != expected:
            raise ValueError("bundle hash differs: " + name)
    for name, expected in TOOL_SHA256.items():
        if sha(bundle / name) != expected:
            raise ValueError("tool does not match the pinned Linux binary")
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    run_id = manifest["run_id"]
    cluster = "bw-astro-" + run_id[:10]
    node, network, kubeconfig = cluster + "-control-plane", cluster + "-net", output / "kubeconfig"
    env = {**os.environ, "DOCKER_CONTEXT": docker_context, "KIND_EXPERIMENTAL_PROVIDER": "docker",
           "KIND_EXPERIMENTAL_DOCKER_NETWORK": network, "KUBECONFIG": str(kubeconfig)}
    command = Commands(output, env)
    docker = ["docker", "--context", docker_context]
    context = json.loads(command(docker + ["context", "inspect", docker_context]))[0]
    info = json.loads(command(docker + ["info", "--format", "{{json .}}",]))
    required = manifest["required_host"]
    if (not context["Endpoints"]["docker"]["Host"].startswith("unix://") or info["OSType"] != "linux"
            or info["Architecture"] not in ("x86_64", "amd64") or info["NCPU"] < required["cpu_count"]
            or info["MemTotal"] < required["memory_bytes"] or shutil.disk_usage(output).free < required["free_disk_bytes"]):
        raise ValueError("host/daemon capacity or local-context contract differs")
    if command(docker + ["ps", "-aq"]).strip():
        raise ValueError("this bootstrap requires a dedicated empty Docker host")
    if network in command(docker + ["network", "ls", "--format", "{{.Name}}"]).split():
        raise ValueError("planned network already exists")
    save(output / "host.json", {"host_id": host_id, "docker": info, "context": context, "manifest": manifest})
    kind, kubectl = str(bundle / "kind-linux-amd64"), str(bundle / "kubectl-linux-amd64")
    kube = [kubectl, "--kubeconfig", str(kubeconfig), "--context", "kind-" + cluster, "--request-timeout=10s"]
    app = json.loads((bundle / "application.json").read_text())["items"]
    control = json.loads((bundle / "control.json").read_text())["items"]
    images = {}
    for row in json.loads((bundle / "image-lock.json").read_text())["images"]:
        for image in row["platforms"]:
            if image["platform"] == "linux/amd64":
                images[image["pinned_ref"]] = {image["manifest_digest"], image["config_digest"]}
    network_id = node_id = None
    process = None
    actor = None
    result = {"passed": False, "cycles": [], "host_id": host_id, "run_id": run_id}

    def items(kind_name, namespace):
        return json.loads(command(kube + ["get", kind_name, "--namespace", namespace, "-o", "json"]))["items"]

    def inventory(folder, phase):
        data = {}
        for resource in ["deployments", "daemonsets", "replicasets", "pods", "services", "endpointslices"]:
            data[resource] = items(resource, manifest["application_namespace"])
        save(folder / (phase + "-inventory.json"), data)
        return data

    try:
        network_id = command(docker + ["network", "create", "--label", "backworld.astronomy=" + run_id, network]).strip()
        save(output / "docker-ownership.json", {"network_id": network_id, "run_id": run_id})
        config = {"kind": "Cluster", "apiVersion": "kind.x-k8s.io/v1alpha4",
                  "networking": {"apiServerAddress": "127.0.0.1"}, "nodes": [{"role": "control-plane", "image": NODE_IMAGE}]}
        save(output / "kind.json", config)
        args = [kind, "create", "cluster", "--name", cluster, "--config", str(output / "kind.json"),
                "--kubeconfig", str(kubeconfig), "--wait", "60s", "--retain"]
        with (output / "kind-create.log").open("w") as log:
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            deadline = time.monotonic() + 300
            while process.poll() is None:
                if time.monotonic() >= min(deadline, command.deadline):
                    raise TimeoutError("cluster bootstrap exceeded300s")
                if node_id is None:
                    raw = command(docker + ["inspect", node], check=False)
                    if raw.strip() and json.loads(raw):
                        actual = json.loads(raw)[0]
                        if actual["Config"]["Labels"].get("io.x-k8s.kind.cluster") != cluster or actual["NetworkSettings"]["Networks"].get(network, {}).get("NetworkID") != network_id:
                            raise ValueError("new node ownership/network differs")
                        node_id = actual["Id"]
                        save(output / "docker-ownership.json", {"network_id": network_id, "node_id": node_id, "run_id": run_id})
                        command(docker + ["update", "--memory", "24g", "--memory-swap", "24g", "--cpus", "6", "--pids-limit", "8192", node_id])
                time.sleep(1)
        if process.returncode or node_id is None:
            raise RuntimeError("cluster did not bootstrap under its caps")
        node_info = json.loads(command(docker + ["inspect", node_id]))[0]
        caps = node_info["HostConfig"]
        if (caps["Memory"] != 24 * 1024 ** 3 or caps["MemorySwap"] != 24 * 1024 ** 3
                or caps["NanoCpus"] != 6 * 1000000000 or caps["PidsLimit"] != 8192):
            raise ValueError("node memory/swap/CPU/PID caps did not take effect")
        save(output / "node.json", node_info)
        nodes = json.loads(command(kube + ["get", "nodes", "-o", "json"]))
        save(output / "nodes.json", nodes)
        if len(nodes["items"]) != 1:
            raise ValueError("the dedicated cluster must have exactly one node")
        node_status = nodes["items"][0]["status"]
        if (node_status["nodeInfo"]["architecture"] != "amd64"
                or not any(c["type"] == "Ready" and c["status"] == "True" for c in node_status["conditions"])
                or min(memory_bytes(node_status["allocatable"]["memory"]), caps["Memory"]) < 5014 * 1024 ** 2):
            raise ValueError("node readiness/architecture or cgroup/allocatable capacity differs")
        previous_uid = None
        for cycle in (1, 2):
            folder = output / f"cycle-{cycle}"
            folder.mkdir()
            actor = KubeControl(command, kube, app + control, folder, run_id)
            actor.create(app)
            namespace = next(o for o in app if o["kind"] == "Namespace")
            uid = actor.receipts[identity(namespace)["key"]]["uid"]
            if uid == previous_uid:
                raise ValueError("reset reused the earlier namespace UID")
            previous_uid = uid
            startup = time.monotonic() + 900
            while True:
                deployments, daemonsets = items("deployments", manifest["application_namespace"]), items("daemonsets", manifest["application_namespace"])
                expected = {o["metadata"]["name"] for o in app if o["kind"] in ("Deployment", "DaemonSet")}
                workloads = deployments + daemonsets
                for workload in workloads:
                    planned = next(o for o in app if o["kind"] == workload["kind"] and o["metadata"]["name"] == workload["metadata"]["name"])
                    owned(workload, planned, run_id, actor.receipts[identity(planned)["key"]]["uid"])
                if {w["metadata"]["name"] for w in workloads} == expected and all(controller_ready(w) for w in workloads):
                    break
                if time.monotonic() >= min(startup, command.deadline):
                    raise TimeoutError("application controllers did not become current/ready")
                time.sleep(3)
            data = inventory(folder, "before")
            for pod in data["pods"]:
                validate_pod(pod, images)
            for service in data["services"]:
                if service["metadata"]["name"] in ("frontend-proxy", "frontend", "cart", "product-catalog", "valkey-cart"):
                    validate_routing(service, data["endpointslices"], data["pods"], data["replicasets"], data["deployments"])
            actor.create(control)
            job_plan = next(o for o in control if o["kind"] == "Job")
            deadline = time.monotonic() + 330
            while True:
                job = actor.get(job_plan)
                owned(job, job_plan, run_id, actor.receipts[identity(job_plan)["key"]]["uid"])
                pods = items("pods", manifest["private_namespace"])
                if job.get("status", {}).get("succeeded") == 1 or job.get("status", {}).get("failed", 0) > 0:
                    break
                if time.monotonic() >= min(deadline, command.deadline):
                    raise TimeoutError("private control did not complete within its deadline")
                time.sleep(1)
            if len(pods) != 1:
                raise ValueError("private job did not create exactly one Pod")
            pod = pods[0]
            save(folder / "job.json", job)
            save(folder / "private-pod.json", pod)
            logs = {}
            for container_name in ("warmup", "check"):
                logs[container_name] = command(kube + ["logs", pod["metadata"]["name"], "--namespace", manifest["private_namespace"], "--container", container_name], check=False)
                (folder / (container_name + ".jsonl")).write_text(logs[container_name])
            validate_pod(pod, images)
            verdict = audit_helper(logs["check"], pod, job, run_id, manifest["application_namespace"], manifest["source_hashes"])
            result["cycles"].append({"cycle": cycle, "namespace_uid": uid, "verdict": verdict})
            inventory(folder, "after")
            command(docker + ["stats", "--no-stream", "--format", "{{json .}}", node_id])
            roots = [next(o for o in control if o["kind"] == "Namespace"), namespace,
                     *[o for o in app if o["kind"] == "ClusterRoleBinding"], *[o for o in app if o["kind"] == "ClusterRole"]]
            cleanup = actor.cleanup(roots)
            save(folder / "cleanup.json", cleanup)
            if not cleanup["ok"]:
                raise RuntimeError("cycle cleanup failed")
            actor = None
        result["passed"] = True
    except BaseException as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        command.deadline = time.monotonic() + 300
        cleanup_errors = []
        if process is not None and process.poll() is None:
            try:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
            except Exception as exc:
                cleanup_errors.append("bootstrap process cleanup: " + str(exc))
        if actor is not None:
            try:
                raw = command(kube + ["get", "pods,deployments,daemonsets,replicasets,services,endpointslices", "--namespace", manifest["application_namespace"], "-o", "json"], check=False)
                (output / "failure-inventory.json").write_text(raw)
            except Exception as exc:
                result["failure_collection_error"] = str(exc)
            try:
                roots = [next(o for o in control if o["kind"] == "Namespace"), next(o for o in app if o["kind"] == "Namespace"),
                         *[o for o in app if o["kind"] == "ClusterRoleBinding"], *[o for o in app if o["kind"] == "ClusterRole"]]
                cleanup = actor.cleanup(roots)
                save(output / "failure-kube-cleanup.json", cleanup)
                if not cleanup["ok"]:
                    cleanup_errors.append("Kubernetes cleanup failed")
            except Exception as exc:
                cleanup_errors.append("Kubernetes cleanup: " + str(exc))
        try:
            if network_id is None and network in command(docker + ["network", "ls", "--format", "{{.Name}}"]).split():
                actual = json.loads(command(docker + ["network", "inspect", network]))[0]
                if actual["Labels"].get("backworld.astronomy") != run_id:
                    raise ValueError("uncertain network creation has different ownership")
                network_id = actual["Id"]
            if node_id is None and network_id and command(docker + ["ps", "-aq", "--filter", "name=^/" + node + "$"]).strip():
                actual = json.loads(command(docker + ["inspect", node]))[0]
                if actual["Config"]["Labels"].get("io.x-k8s.kind.cluster") != cluster or actual["NetworkSettings"]["Networks"].get(network, {}).get("NetworkID") != network_id:
                    raise ValueError("uncertain node creation has different ownership")
                node_id = actual["Id"]
            if node_id:
                if command(docker + ["ps", "-aq", "--filter", "id=" + node_id]).strip():
                    actual = json.loads(command(docker + ["inspect", node_id]))[0]
                    if actual["Id"] != node_id or actual["Config"]["Labels"].get("io.x-k8s.kind.cluster") != cluster:
                        raise ValueError("node identity/owner changed before cleanup")
                    command(docker + ["rm", "--force", node_id])
            if network_id:
                actual = json.loads(command(docker + ["network", "inspect", network_id]))[0]
                if actual["Id"] != network_id or actual["Labels"].get("backworld.astronomy") != run_id or actual.get("Containers"):
                    raise ValueError("network owner changed or it still has containers")
                command(docker + ["network", "rm", network_id])
            remaining = command(docker + ["ps", "-aq", "--filter", "label=io.x-k8s.kind.cluster=" + cluster]).split()
            if remaining:
                cleanup_errors.append("owned nodes remain: " + str(remaining))
            if network in command(docker + ["network", "ls", "--format", "{{.Name}}"]).split():
                cleanup_errors.append("owned network remains")
        except Exception as exc:
            cleanup_errors.append("Docker cleanup: " + str(exc))
        result["cleanup"] = {"ok": not cleanup_errors, "errors": cleanup_errors}
        result["passed"] &= not cleanup_errors
        result["cloud_host_cleanup"] = "external provider/host owner must verify; not a provider lifecycle tool"
        save(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--inputs", type=Path, required=True)
    prep.add_argument("--tools", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--bundle", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--docker-context", required=True)
    execute.add_argument("--host-id", required=True)
    execute.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(args.inputs.resolve(), args.tools.resolve(), args.output.resolve()), indent=2))
    else:
        if not args.execute:
            parser.error("run requires --execute on an explicitly owned approved host")
        def interrupted(signum, frame):
            raise RuntimeError("host runner interrupted")
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        result = run(args.bundle.resolve(), args.output.resolve(), args.docker_context, args.host_id)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
