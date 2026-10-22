import copy
import hashlib
import importlib.util
import json
import math
import sys
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from service_operator import hotel


WORKLOAD_BLOB = "b38ebb60ecbedcbb80eafd3a940a770b2b815983"
PORTS = {"geo": 8083, "rate": 8084, "profile": 8081,
         "mongodb-geo": 27017, "mongodb-rate": 27017, "mongodb-profile": 27017,
         "memcached-rate": 11211, "memcached-profile": 11211}
VULNERABLE = {"rate": {"RATE_BACKEND_QPS_LIMIT": "20", "RATE_QUEUE_CAPACITY": "256"},
              "search": {"RATE_RPC_TIMEOUT_MS": "750", "RATE_RPC_MAX_ATTEMPTS": "3",
                         "RATE_RPC_INITIAL_BACKOFF_MS": "50", "RATE_RPC_BACKOFF_MULTIPLIER": "2.0", "RATE_RPC_JITTER": "0.2"}}
MITIGATED = {"rate": {"RATE_BACKEND_QPS_LIMIT": "50", "RATE_QUEUE_CAPACITY": "256"},
             "search": {**VULNERABLE["search"], "RATE_RPC_TIMEOUT_MS": "1000", "RATE_RPC_MAX_ATTEMPTS": "2"}}
CATALOG_JS = 'print(JSON.stringify(db.getSiblingDB("profile-db").hotels.find({},{_id:0,id:1,name:1,address:1}).sort({id:1}).toArray()));'


def profile(namespace, aliases, init_scripts):
    objects = hotel.kubernetes_profile(namespace, aliases, probes=False)
    definitions = [(name, "app", "384Mi", ["/go/bin/" + name]) for name in ("geo", "rate", "profile")]
    definitions += [("mongodb-" + name, "mongo", "512Mi",
                     ["mongod", "--bind_ip_all", "--wiredTigerCacheSizeGB", "0.25", "--quiet"])
                    for name in ("geo", "rate", "profile")]
    definitions += [("memcached-" + name, "memcached", "64Mi", ["memcached", "-m", "32", "-t", "2"])
                    for name in ("rate", "profile")]
    for name, role, memory, command in definitions:
        obj = hotel.deployment(name, role, aliases[role], memory, command)
        obj["metadata"]["namespace"] = namespace
        pod = obj["spec"]["template"]["spec"]
        container = pod["containers"][0]
        container["readinessProbe"] = {"tcpSocket": {"port": PORTS[name]}, "periodSeconds": 1,
                                       "timeoutSeconds": 1, "failureThreshold": 60}
        if name in ("mongodb-geo", "mongodb-rate"):
            key = name.removeprefix("mongodb-")
            container["command"].insert(0, "/usr/local/bin/docker-entrypoint.sh")
            container["command"].append("--auth")
            container["volumeMounts"].append({"name": "init", "mountPath": "/docker-entrypoint-initdb.d", "readOnly": True})
            pod["volumes"].append({"name": "init", "configMap": {"name": "mongo-init-" + key}})
            objects.append({"apiVersion": "v1", "kind": "ConfigMap",
                "metadata": {"name": "mongo-init-" + key, "namespace": namespace},
                "data": {"init.sh": init_scripts[key]}})
        objects += [obj, {"apiVersion": "v1", "kind": "Service", "metadata": {"name": name, "namespace": namespace},
            "spec": {"selector": {"io.kompose.service": name}, "ports": [{"name": "rpc", "port": PORTS[name], "targetPort": PORTS[name]}]}}]
    for obj in objects:
        name = obj["metadata"]["name"]
        if obj["kind"] == "Deployment" and name in VULNERABLE:
            container = obj["spec"]["template"]["spec"]["containers"][0]
            container["env"] += [{"name": k, "value": v} for k, v in VULNERABLE[name].items()]
        if obj["kind"] == "Service" and name in ("search", "rate"):
            obj["spec"]["ports"][0]["name"] = "rpc"
            port = 9092 if name == "search" else 9091
            obj["spec"]["ports"].append({"name": "metrics", "port": port, "targetPort": port})
    return objects


def select_hotel(response, catalog):
    if response.get("status") != 200 or response.get("error"):
        raise ValueError("search did not return a successful response")
    elapsed, deadline = response.get("elapsed_s"), response.get("deadline_s")
    if (not all(type(v) in (int, float) and math.isfinite(v) for v in (elapsed, deadline))
            or not 0 <= elapsed <= deadline or deadline <= 0):
        raise ValueError("search response was late")
    body = response.get("body")
    if not isinstance(body, dict) or body.get("type") != "FeatureCollection":
        raise ValueError("search did not return a feature collection")
    features = body.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("search returned no hotels")
    seen = set()
    for feature in features:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError("search returned an invalid feature")
        key = feature.get("id")
        if not isinstance(key, str) or key not in catalog or key in seen:
            raise ValueError("search returned an unknown or duplicate hotel")
        seen.add(key)
        point = feature.get("geometry", {})
        if not isinstance(point, dict):
            raise ValueError("search returned an invalid location")
        coordinates = point.get("coordinates")
        if point.get("type") != "Point" or not isinstance(coordinates, list) or len(coordinates) != 2:
            raise ValueError("search returned an invalid location")
        expected = [catalog[key]["address"]["lon"], catalog[key]["address"]["lat"]]
        if any(type(a) not in (float, int) or not math.isfinite(a) or abs(a - b) > 0.0001
               for a, b in zip(coordinates, expected)):
            raise ValueError("search returned the wrong hotel location")
        properties = feature.get("properties")
        if not isinstance(properties, dict) or properties.get("name") != catalog[key]["name"]:
            raise ValueError("search returned the wrong hotel name")
    return features[0]["id"]


def delta(before, after, key):
    if not all(type(s.get(key)) in (int, float) and math.isfinite(s[key]) for s in (before, after)):
        raise ValueError("required counter is missing or invalid: " + key)
    value = after[key] - before[key]
    if value < 0:
        raise ValueError("counter reset inside a measurement: " + key)
    return value


def window(events, start, end, catalog, before, after):
    events = [event for event in events if event["kind"] == "search" and start <= event["started_s"] < end]
    valid = 0
    for event in events:
        try:
            select_hotel(event.get("response", {}), catalog)
            valid += 1
        except ValueError:
            pass
    search = delta(before, after, "search_requests_total")
    attempts = delta(before, after, "search_rate_attempts_total")
    return {"duration_s": end - start, "actual_started": len(events), "actual_rate": len(events) / (end - start),
            "completed": sum("finished_s" in event for event in events), "valid": valid,
            "success_rate": valid / len(events) if events else 0,
            "search_requests": search, "attempts_per_request": attempts / search if search else None,
            "queue_before": before.get("rate_queue_depth"), "queue_after": after.get("rate_queue_depth"),
            "expired_work": delta(before, after, "rate_requests_expired_total"),
            "backend_completed": delta(before, after, "rate_requests_completed_total")}


def healthy(sample, initial=False):
    ratio = sample["attempts_per_request"]
    return ((6.8 if initial else 6.4) <= sample["actual_rate"] <= 9.6
            and sample["completed"] == sample["actual_started"]
            and sample["completed"] >= 8 * sample["duration_s"] * 0.75
            and sample["search_requests"] >= 8 * sample["duration_s"] * 0.75
            and sample["success_rate"] >= 0.9 and ratio is not None and ratio <= (1.2 if initial else 2.1)
            and sample["queue_after"] is not None and 0 <= sample["queue_after"] <= 5)


def collapsed(sample):
    ratio = sample["attempts_per_request"]
    return (6.4 <= sample["actual_rate"] <= 9.6 and sample["completed"] == sample["actual_started"]
            and sample["queue_before"] is not None and sample["queue_after"] is not None
            and sample["queue_after"] >= 20 and sample["queue_after"] + 5 >= sample["queue_before"]
            and ratio is not None and ratio >= 2 and sample["success_rate"] <= 0.5 and sample["expired_work"] > 0)


def safe_policy(metrics):
    values = [metrics.get(key) for key in ("rate_backend_qps_limit", "rate_queue_capacity")]
    return (all(type(v) in (int, float) and math.isfinite(v) for v in values)
            and 0 < values[0] <= 500 and 0 < values[1] <= 256)


class RecordedRequests:
    RequestException = requests.RequestException

    def __init__(self, output):
        self.output = output
        self.lock = threading.Lock()
        self.events = []

    def write(self, stage, event):
        with self.output.open("a") as handle:
            handle.write(json.dumps({"stage": stage, "event": event}) + "\n")

    def get(self, url, **kwargs):
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path not in ("/hotels", "/metrics"):
            raise ValueError("workload URL escaped the declared local tunnels")
        event = {"kind": "metrics" if parsed.path == "/metrics" else "search", "url": url,
                 "started_s": time.monotonic(), "started_unix_s": time.time()}
        with self.lock:
            event["id"] = len(self.events)
            self.events.append(event)
            self.write("started", event)
        receipt = None
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.get(url, **kwargs)
            try:
                body = response.json()
            except requests.JSONDecodeError:
                body = response.text
            receipt = {"status": response.status_code, "body": body,
                                 "elapsed_s": time.monotonic() - event["started_s"],
                                 "deadline_s": kwargs["timeout"]}
            return response
        except Exception as exc:
            receipt = {"status": None, "body": None, "error": f"{type(exc).__name__}: {exc}",
                                 "elapsed_s": time.monotonic() - event["started_s"], "deadline_s": kwargs["timeout"]}
            raise
        finally:
            with self.lock:
                event["response"] = receipt
                event["finished_s"] = time.monotonic()
                self.write("completed", event)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.events)


def load_workload(path, recorder):
    body = path.read_bytes()
    if hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest() != WORKLOAD_BLOB:
        raise ValueError("workload source differs from the pinned revision")
    spec = importlib.util.spec_from_file_location("backworld_pinned_search_workload", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.requests = SimpleNamespace(get=recorder.get, RequestException=requests.RequestException)
    return module.HotelSearchWorkload
