import argparse
import hashlib
import http.client
import json
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from service_operator import workflow
from service_operator.workflow import check_booking


IMAGES = {
    "app": {
        "ref": "ghcr.io/sregym/hotel-reservation@sha256:7918331dfec6fa3d71d0af5d92ff3696f85a6f620d876c7bf78b85d852d7c299",
        "platform": "linux/amd64", "observed_tag": "latest",
    },
    "mongo": {
        "ref": "mongo@sha256:aa3f99f22e195adee9071152b4c87536f795c21ca233eeb3a28ca33af72ec1d6",
        "platform": "linux/arm64/v8", "observed_tag": "4.4.6",
    },
    "consul": {
        "ref": "hashicorp/consul@sha256:fbed4e2bd2d25f2aefb5a6ee68ead0b7fdb34bad14ce62889063bfb666b02c87",
        "platform": "linux/arm64", "observed_tag": "1.22.3",
    },
    "memcached": {
        "ref": "memcached@sha256:b263323af06e550df9becd8c2cf8450400be7ae4381392593565dc8247bb230f",
        "platform": "linux/arm64/v8", "observed_tag": "1.6.39-alpine",
    },
    "jaeger": {
        "ref": "jaegertracing/all-in-one@sha256:84d2b9664cea496af47fb58077319b55569542804884041a87b175475e183628",
        "platform": "linux/arm64", "observed_tag": "1.57",
    },
}
SEED = {"hotelId": "4", "customerName": "Alice", "inDate": "2015-04-09",
        "outDate": "2015-04-10", "number": 1}
SNAPSHOT_JS = (
    'var d=db.getSiblingDB("reservation-db");'
    'function rows(c){return d.getCollection(c).find().sort({_id:1}).toArray().map(function(r){'
    'if(!(r._id instanceof ObjectId)){throw new Error("unexpected id type");}'
    'r._id=r._id.str;return r;});}'
    'print(JSON.stringify({reservations:rows("reservation"),capacity:rows("number")}));'
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def compose_config(run_id):
    def service(image, memory, **extra):
        return {
            "image": IMAGES[image]["ref"], "platform": IMAGES[image]["platform"],
            "mem_limit": memory, "cpus": 1, "pids_limit": 128, "restart": "no",
            "labels": {"backworld.service-workflow": run_id},
            "logging": {"driver": "json-file", "options": {"max-size": "5m", "max-file": "1"}},
            **extra,
        }

    def health(command):
        return {"test": ["CMD", *command], "interval": "2s", "timeout": "3s", "retries": 30}

    services = {
        "consul": service("consul", "256m", healthcheck=health(["consul", "members"])),
        "jaeger": service("jaeger", "192m", environment={"MEMORY_MAX_TRACES": "1000"}),
        "memcached-reserve": service("memcached", "64m", command=["memcached", "-m", "32", "-t", "2"]),
    }
    volumes = {}
    for name in ("user", "reservation"):
        volumes[name + "-db"] = {}
        volumes[name + "-config"] = {}
        services["mongodb-" + name] = service(
            "mongo", "512m",
            command=["mongod", "--bind_ip_all", "--wiredTigerCacheSizeGB", "0.25", "--quiet"],
            volumes=[name + "-db:/data/db", name + "-config:/data/configdb"],
            healthcheck=health(["mongo", "--quiet", "--eval", "quit(db.runCommand({ping:1}).ok ? 0 : 1)"]),
        )
    for name in ("user", "reservation", "frontend"):
        dependencies = {"consul": {"condition": "service_healthy"}, "jaeger": {"condition": "service_started"}}
        if name != "frontend":
            dependencies["mongodb-" + name] = {"condition": "service_healthy"}
        if name == "reservation":
            dependencies["memcached-reserve"] = {"condition": "service_started"}
        services[name] = service(
            "app", "384m", entrypoint=["/go/bin/" + name], depends_on=dependencies,
            environment={"LOG_LEVEL": "warn", "JAEGER_SAMPLE_RATIO": "0.01", "GOMAXPROCS": "2"},
        )
    services["frontend"]["ports"] = [{"target": 5000, "host_ip": "127.0.0.1", "protocol": "tcp"}]
    return {"services": services, "volumes": volumes}


def request(url, deadline=5):
    started = time.monotonic()
    result = {"status": None, "body": None, "deadline_s": deadline}
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=deadline) as response:
            result["status"] = response.status
            result["body"] = json.loads(response.read(65536))
    except (OSError, ValueError, http.client.HTTPException, urllib.error.URLError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, urllib.error.HTTPError):
            result["status"] = exc.code
    result["elapsed_s"] = time.monotonic() - started
    return result


def audit(output, context):
    output.mkdir(parents=True, exist_ok=False)
    docker = ["docker", "--context", context]
    run_id = "bw-hotel-" + uuid.uuid4().hex[:10]
    started = time.monotonic()
    result = {"run_id": run_id, "status": "running", "cycles": []}
    command_count = 0
    source_hashes = {}
    for source in (Path(__file__), Path(workflow.__file__)):
        data = source.read_bytes()
        (output / source.name).write_bytes(data)
        source_hashes[source.name] = hashlib.sha256(data).hexdigest()

    def command(args, timeout=30):
        nonlocal command_count
        command_count += 1
        prefix = output / f"command-{command_count:03d}"
        begin = time.monotonic()
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
            prefix.with_suffix(".log").write_text(proc.stdout + proc.stderr)
            write_json(prefix.with_suffix(".json"), {
                "args": args, "elapsed_s": time.monotonic() - begin, "returncode": proc.returncode,
            })
        except subprocess.TimeoutExpired as exc:
            write_json(prefix.with_suffix(".json"), {
                "args": args, "elapsed_s": time.monotonic() - begin, "error": "timeout",
            })
            prefix.with_suffix(".log").write_bytes((exc.stdout or b"") + (exc.stderr or b""))
            raise
        if proc.returncode:
            raise RuntimeError(f"command failed; see {prefix.with_suffix('.log')}")
        return proc.stdout.strip()

    def resources(project):
        label = "label=com.docker.compose.project=" + project
        return {
            "containers": command(docker + ["ps", "-aq", "--filter", label]).split(),
            "networks": command(docker + ["network", "ls", "-q", "--filter", label]).split(),
            "volumes": command(docker + ["volume", "ls", "-q", "--filter", label]).split(),
        }

    try:
        manifest = {
            "scope": "booking-only compose preflight; no Kubernetes fault, agent or performance claim",
            "source_revision": "2b2f9c6c2e97c44abbfcc44af1cf2f994bbb04f8",
            "source_evidence": "43 inspected source/config files match; binary build is not attested",
            "images": IMAGES, "context": context,
            "harness_sha256": source_hashes,
            "docker": json.loads(command(docker + ["info", "--format", "{{json .}}"])),
            "compose_version": command(docker + ["compose", "version", "--short"]),
            "protocol": {
                "cycles": 2, "requests_per_cycle": 2, "request_deadline_s": 5,
                "settle_s": 1, "readiness_deadline_s": 60,
                "operation": "sequential one-night booking; same hotel/dates to exercise warm cache",
                "success": "timely acknowledgement, exactly one matching write, all prior rows and capacity preserved",
                "uncertain_write": "record database state; never retry booking",
                "reset": "destroy owned containers/network/volumes and verify fresh seed on next cycle",
                "limits": "8 containers, total memory caps 2688 MiB; only frontend published on loopback",
            },
        }
        write_json(output / "manifest.json", manifest)
        if manifest["docker"]["Architecture"] not in ("aarch64", "arm64"):
            raise RuntimeError("this preflight pins ARM dependencies and an emulated AMD64 app")
        for role, image in IMAGES.items():
            print("pulling " + role, flush=True)
            command(docker + ["pull", "--platform", image["platform"], image["ref"]], timeout=240)
            write_json(output / (role + "-image.json"), json.loads(command(docker + ["image", "inspect", image["ref"]])))
        for cycle in (1, 2):
            folder = output / f"cycle-{cycle}"
            folder.mkdir()
            project = run_id + f"-{cycle}"
            config_path = folder / "compose.json"
            write_json(config_path, compose_config(run_id))
            compose = docker + ["compose", "--project-name", project, "--env-file", "/dev/null", "--file", str(config_path)]
            entry = {"cycle": cycle, "project": project, "bookings": [], "cleanup_ok": False}
            result["cycles"].append(entry)

            def snapshot():
                return json.loads(command(compose + ["exec", "-T", "mongodb-reservation", "mongo", "--quiet", "--eval", SNAPSHOT_JS]))

            try:
                print(f"starting cycle {cycle}", flush=True)
                launch = time.monotonic()
                command(compose + ["up", "-d", "--no-build", "--pull", "never"], timeout=150)
                address = command(compose + ["port", "frontend", "5000"])
                if not address.startswith("127.0.0.1:"):
                    raise RuntimeError("frontend must be bound to loopback")
                base = "http://" + address
                readiness = []
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    login = request(base + "/user?username=Cornell_1&password=1111111111", deadline=2)
                    registered = command(compose + ["exec", "-T", "consul", "consul", "catalog", "services"]).split()
                    readiness.append({"login": login, "registered": registered})
                    write_json(folder / "readiness.json", readiness)
                    if login.get("body") == {"message": "Login successfully!"} and "srv-reservation" in registered:
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError("booking dependencies never became ready")
                entry["ready_s"] = time.monotonic() - launch
                initial = snapshot()
                write_json(folder / "initial.json", initial)
                seed_rows = [{key: value for key, value in row.items() if key != "_id"}
                             for row in initial["reservations"]]
                capacity = {row["hotelId"]: row["numberOfRoom"] for row in initial["capacity"]}
                expected_capacity = {str(i): (300 if i % 3 == 1 else 250 if i % 3 == 2 else 200)
                                     if i >= 7 else 200 for i in range(1, 81)}
                entry["fresh_seed"] = (seed_rows == [SEED] and len(initial["capacity"]) == 80
                                       and capacity == expected_capacity)
                if not entry["fresh_seed"]:
                    raise RuntimeError("initial state differs from the expected fresh seed")
                before = initial
                for phase in ("cold", "warm"):
                    expected = {
                        "hotelId": "1", "customerName": f"{run_id}-{cycle}-{phase}",
                        "inDate": "2026-11-01", "outDate": "2026-11-02", "number": 1,
                    }
                    query = urllib.parse.urlencode({**expected, "username": "Cornell_1", "password": "1111111111"})
                    response = request(base + "/reservation?" + query)
                    event = {"phase": phase, "expected": expected, "response": response, "before": before}
                    write_json(folder / (phase + ".json"), event)
                    booking = {"phase": phase, "response": response, "verdict": None}
                    entry["bookings"].append(booking)
                    write_json(output / "summary.json", result)
                    immediate = snapshot()
                    event["immediate"] = immediate
                    write_json(folder / (phase + ".json"), event)
                    time.sleep(1)
                    after = snapshot()
                    verdict = check_booking(before["reservations"], after["reservations"], expected, response)
                    verdict["capacity_preserved"] = before["capacity"] == after["capacity"]
                    verdict["stable_after_response"] = immediate == after
                    verdict["accepted"] &= verdict["capacity_preserved"] and verdict["stable_after_response"]
                    event.update({"verdict": verdict, "after": after})
                    write_json(folder / (phase + ".json"), event)
                    booking["verdict"] = verdict
                    print(f"cycle {cycle} {phase}: {verdict['outcome']}", flush=True)
                    if not verdict["accepted"]:
                        raise RuntimeError("booking verification failed; no write retry")
                    before = after
                ids = command(compose + ["ps", "-q"]).split()
                write_json(folder / "containers.json", json.loads(command(docker + ["inspect", *ids])))
                (folder / "stats.txt").write_text(command(docker + ["stats", "--no-stream", *ids], timeout=30) + "\n")
            finally:
                try:
                    (folder / "services.log").write_text(command(compose + ["logs", "--no-color", "--tail", "250"]) + "\n")
                finally:
                    command(compose + ["down", "--volumes", "--timeout", "5"], timeout=60)
                    remaining = resources(project)
                    entry["remaining_resources"] = remaining
                    entry["cleanup_ok"] = not any(remaining.values())
                    write_json(output / "summary.json", result)
                    if not entry["cleanup_ok"]:
                        raise RuntimeError("owned resources remain after cleanup")
        result["status"] = "passed"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_s"] = time.monotonic() - started
        write_json(output / "summary.json", result)
        print(json.dumps({key: value for key, value in result.items() if key != "cycles"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", default="desktop-linux")
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    audit(args.output.resolve(), args.context)


if __name__ == "__main__":
    main()
