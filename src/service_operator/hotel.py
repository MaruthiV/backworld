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

CLIENT_IMAGE = {
    "ref": "python@sha256:739ba32ae445e8d58f3d90feb85f83bebc8346f8dd280fa1eb5848f4ff1ed163",
    "platform": "linux/arm64/v8", "observed_tag": "3.12-slim-bookworm",
}
PORTS = {"consul": 8500, "jaeger": 6831, "memcached-reserve": 11211,
         "mongodb-user": 27017, "mongodb-reservation": 27017,
         "user": 8086, "reservation": 8087, "search": 8082, "frontend": 5000,
         "false-response": 5000}
FAKE_SERVER = (
    'import http.server,json;'
    'body=json.dumps({"message":"Reserve successfully!"}).encode();'
    'handler=type("Handler",(http.server.BaseHTTPRequestHandler,),'
    '{"do_GET":lambda s:(s.send_response(200),s.send_header("Content-Type","application/json"),'
    's.send_header("Content-Length",str(len(body))),s.end_headers(),s.wfile.write(body)),'
    '"log_message":lambda *a:None});'
    'http.server.HTTPServer(("0.0.0.0",5000),handler).serve_forever()'
)


def fresh_seed(snapshot):
    reservations = [{k: v for k, v in row.items() if k != "_id"} for row in snapshot["reservations"]]
    capacity = {row["hotelId"]: row["numberOfRoom"] for row in snapshot["capacity"]}
    expected = {str(i): (300 if i % 3 == 1 else 250 if i % 3 == 2 else 200)
                if i >= 7 else 200 for i in range(1, 81)}
    return reservations == [SEED] and len(snapshot["capacity"]) == 80 and capacity == expected


def deployment(name, role, alias, memory, command=None):
    container = {
        "name": name, "image": alias, "imagePullPolicy": "Never",
        "resources": {"requests": {"cpu": "50m", "memory": "32Mi"},
                      "limits": {"cpu": "1", "memory": memory}},
    }
    if command:
        container["command"] = command
    if name in PORTS:
        container["readinessProbe"] = {"tcpSocket": {"port": 14269 if role == "jaeger" else PORTS[name]}, "periodSeconds": 1,
                                       "timeoutSeconds": 1, "failureThreshold": 60}
    if role == "app":
        container["env"] = [{"name": key, "value": value} for key, value in
                            {"LOG_LEVEL": "warn", "JAEGER_SAMPLE_RATIO": "0.01", "GOMAXPROCS": "2"}.items()]
    if role == "mongo":
        container["volumeMounts"] = [{"name": "data", "mountPath": "/data/db"},
                                      {"name": "config", "mountPath": "/data/configdb"}]
    if role == "jaeger":
        container["env"] = [{"name": "MEMORY_MAX_TRACES", "value": "1000"}]
    spec = {"automountServiceAccountToken": False, "containers": [container]}
    if role == "mongo":
        spec["volumes"] = [{"name": "data", "emptyDir": {}}, {"name": "config", "emptyDir": {}}]
    labels = {"io.kompose.service": name}
    return {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": name},
            "spec": {"replicas": 1, "selector": {"matchLabels": labels},
                     "template": {"metadata": {"labels": labels}, "spec": spec}}}


def kubernetes_profile(namespace, aliases):
    objects = []
    definitions = [
        ("consul", "consul", "256Mi", None), ("jaeger", "jaeger", "192Mi", None),
        ("memcached-reserve", "memcached", "64Mi", ["memcached", "-m", "32", "-t", "2"]),
        *[("mongodb-" + name, "mongo", "512Mi",
           ["mongod", "--bind_ip_all", "--wiredTigerCacheSizeGB", "0.25", "--quiet"])
          for name in ("user", "reservation")],
        *[(name, "app", "384Mi", ["/go/bin/" + name]) for name in ("user", "reservation", "search", "frontend")],
        ("client", "client", "128Mi", ["python", "-c", "import time; time.sleep(3600)"]),
        ("false-response", "client", "128Mi", ["python", "-c", FAKE_SERVER]),
    ]
    for name, role, memory, command in definitions:
        item = deployment(name, role, aliases[role], memory, command)
        item["metadata"]["namespace"] = namespace
        objects.append(item)
        if name in PORTS:
            port = PORTS[name]
            objects.append({"apiVersion": "v1", "kind": "Service",
                            "metadata": {"name": name, "namespace": namespace},
                            "spec": {"selector": {"io.kompose.service": name},
                                     "ports": [{"port": port, "targetPort": port,
                                                "protocol": "UDP" if name == "jaeger" else "TCP"}]}})
    return objects

