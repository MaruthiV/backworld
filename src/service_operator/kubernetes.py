import hashlib
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path


NODE_IMAGE = "kindest/node:v1.34.11@sha256:44e222ee2132dab25ff87301682f89eb82c7880ea3a1bf543bfe9708fd08d67d"
KIND_SHA256 = "0c8c7dbe5e23594a198b786c4bc13dacc101fa6196b0cb0b23a1ca44e61f4b4f"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class LocalCluster:
    def __init__(self, output, kind, context="desktop-linux"):
        self.output = output
        self.kind = str(kind.resolve())
        if hashlib.sha256(kind.read_bytes()).hexdigest() != KIND_SHA256:
            raise ValueError("kind binary differs from the pinned macOS ARM release")
        self.name = "bw-selector-" + uuid.uuid4().hex[:10]
        self.node = self.name + "-control-plane"
        self.network = self.name + "-net"
        self.kubeconfig = output / "kubeconfig"
        self.docker = ["docker", "--context", context]
        self.env = {**os.environ, "DOCKER_CONTEXT": context, "KIND_EXPERIMENTAL_PROVIDER": "docker",
                    "KIND_EXPERIMENTAL_DOCKER_NETWORK": self.network, "KUBECONFIG": str(self.kubeconfig)}
        self.count = 0
        self.command_lock = threading.Lock()
        self.network_created = False
        self.process = None
        self.aliases = {}
        self.image_records = {}

    def command(self, args, timeout=30, input=None, check=True):
        with self.command_lock:
            self.count += 1
            prefix = self.output / f"command-{self.count:04d}"
        started = time.monotonic()
        record = {"args": args, "started_unix_s": time.time()}
        try:
            result = subprocess.run(args, input=input, capture_output=True, text=True,
                                    timeout=timeout, env=self.env)
            record["returncode"] = result.returncode
            prefix.with_suffix(".log").write_text(result.stdout + result.stderr)
        except subprocess.TimeoutExpired as exc:
            record["error"] = "timeout"
            prefix.with_suffix(".log").write_bytes((exc.stdout or b"") + (exc.stderr or b""))
            raise
        finally:
            record["elapsed_s"] = time.monotonic() - started
            save(prefix.with_suffix(".json"), record)
        if check and result.returncode:
            raise RuntimeError(f"command failed; see {prefix.with_suffix('.log')}")
        return result.stdout.strip()

    def kube(self, args, input=None, timeout=30, check=True):
        prefix = self.docker + ["exec"] + (["-i"] if input is not None else [])
        return self.command(prefix + [self.node, "kubectl", "--kubeconfig", "/etc/kubernetes/admin.conf", *args],
                            timeout=timeout, input=input, check=check)

    def apply(self, objects):
        return self.kube(["apply", "-f", "-"], input=json.dumps({"apiVersion": "v1", "kind": "List", "items": objects}))

    def start(self):
        self.command(self.docker + ["network", "create", "--label", "backworld.service-kube=" + self.name, self.network])
        self.network_created = True
        config = {"kind": "Cluster", "apiVersion": "kind.x-k8s.io/v1alpha4",
                  "networking": {"apiServerAddress": "127.0.0.1"},
                  "nodes": [{"role": "control-plane", "image": NODE_IMAGE}]}
        save(self.output / "kind.json", config)
        args = [self.kind, "create", "cluster", "--name", self.name, "--config", str(self.output / "kind.json"),
                "--kubeconfig", str(self.kubeconfig), "--wait", "120s", "--retain"]
        with (self.output / "kind-create.log").open("w") as log:
            self.process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=self.env)
            deadline = time.monotonic() + 180
            limited = False
            while self.process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("cluster creation deadline")
                if not limited:
                    probe = subprocess.run(self.docker + ["inspect", self.node], capture_output=True,
                                           text=True, timeout=10, env=self.env)
                    if probe.returncode == 0:
                        info = json.loads(probe.stdout)[0]
                        if info["Config"]["Labels"].get("io.x-k8s.kind.cluster") != self.name:
                            raise RuntimeError("unexpected node ownership")
                        self.command(self.docker + ["update", "--memory", "4g", "--memory-swap", "4g",
                                                   "--cpus", "4", "--pids-limit", "2048", self.node])
                        limited = True
                time.sleep(0.5)
        if self.process.returncode or not limited:
            raise RuntimeError("cluster startup failed; see kind-create.log")
        save(self.output / "node-inspect.json", json.loads(self.command(self.docker + ["inspect", self.node])))
        save(self.output / "nodes.json", json.loads(self.kube(["get", "nodes", "-o", "json"])))

    def load_images(self, images):
        for role, image in images.items():
            print("loading " + role, flush=True)
            local_id = image.get("local_image_id")
            if local_id:
                if image["ref"] != local_id or not local_id.startswith("sha256:") or len(local_id) != 71:
                    raise ValueError("local image must use its exact content ID")
                info = json.loads(self.command(self.docker + ["image", "inspect", local_id]))[0]
                if info["Id"] != local_id or info["Architecture"] != image["platform"].split("/")[1]:
                    raise ValueError("local image content or platform differs")
            else:
                self.command(self.docker + ["pull", "--platform", image["platform"], image["ref"]], timeout=240)
            alias = f"backworld.local/{role}:{self.name}"
            self.command(self.docker + ["tag", image["ref"], alias])
            self.aliases[role] = alias
            original = json.loads(self.command(self.docker + ["image", "inspect", image["ref"]]))[0]
            tagged = json.loads(self.command(self.docker + ["image", "inspect", alias]))[0]
            if original["Id"] != tagged["Id"]:
                raise RuntimeError("local alias does not match pinned image")
            self.image_records[role] = {"source": image, "alias": alias, "image_id": original["Id"],
                                        "architecture": original["Architecture"]}
            save(self.output / "image-aliases.json", self.image_records)
            self.command([self.kind, "load", "docker-image", alias, "--name", self.name], timeout=180)
            imported = json.loads(self.command(self.docker + ["exec", self.node, "crictl", "inspecti", alias]))
            save(self.output / (role + "-runtime-image.json"), imported)
            if alias not in imported["status"].get("repoTags", []):
                raise RuntimeError("imported image is not available to Kubernetes: " + alias)

    def close(self):
        errors = []
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=15)
        try:
            self.command([self.kind, "export", "logs", str(self.output / "cluster-logs"), "--name", self.name], timeout=60)
        except Exception as exc:
            errors.append("log export: " + str(exc))
        try:
            self.command([self.kind, "delete", "cluster", "--name", self.name,
                          "--kubeconfig", str(self.kubeconfig)], timeout=90)
            if self.network_created:
                info = json.loads(self.command(self.docker + ["network", "inspect", self.network]))[0]
                if info.get("Labels", {}).get("backworld.service-kube") != self.name or info.get("Containers"):
                    raise RuntimeError("network ownership changed or it is not empty")
                self.command(self.docker + ["network", "rm", self.network])
        except Exception as exc:
            errors.append("cluster cleanup: " + str(exc))
        for alias in self.aliases.values():
            try:
                self.command(self.docker + ["image", "rm", alias])
            except Exception as exc:
                errors.append("tag cleanup: " + str(exc))
        remaining = self.command(self.docker + ["ps", "-aq", "--filter", "label=io.x-k8s.kind.cluster=" + self.name])
        result = {"ok": not errors and not remaining, "errors": errors, "remaining_nodes": remaining.split()}
        save(self.output / "cleanup.json", result)
        return result
