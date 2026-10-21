import json
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from service_operator import http_probe
except ModuleNotFoundError:
    import http_probe


class Traffic:
    def __init__(self, base_url, output, count=12, interval=3):
        self.base_url, self.output = base_url, output
        self.count, self.interval = count, interval
        self.started = None
        self.ended = None
        self.declared_complete = None
        self.events = []
        self.journal = []
        self.error = None
        self.serial = threading.Lock()
        self.records = threading.Lock()

    def record(self, stage, event):
        row = json.loads(json.dumps({"stage": stage, "event": event}))
        self.journal.append(row)
        line = json.dumps(row)
        with self.output.open("a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def probe(self, origin, planned=None):
        offered = time.monotonic() if planned is None else planned
        with self.serial:
            expected = {"hotelId": "1", "customerName": "req-" + uuid.uuid4().hex,
                        "inDate": "2026-11-01", "outDate": "2026-11-02", "number": 1}
            event = {"origin": origin, "expected": expected, "offered_monotonic_s": offered,
                     "started_monotonic_s": time.monotonic(), "started_unix_s": time.time()}
            with self.records:
                self.events.append(event)
                self.record("offered", event)
            query = urllib.parse.urlencode({**expected, "username": "Cornell_1", "password": "1111111111"})
            response = http_probe.request(self.base_url + "/reservation?" + query, 1)
            with self.records:
                event.update({"response": response, "acknowledgement_monotonic_s": time.monotonic()})
                self.record("completed", event)
            return event

    def start(self):
        if self.started is not None:
            raise ValueError("workload already started")
        self.started = time.monotonic()
        threading.Thread(target=self.run, daemon=True).start()
        return {"started_monotonic_s": self.started}

    def run(self):
        try:
            for slot in range(self.count):
                planned = self.started + slot * self.interval
                time.sleep(max(0, planned - time.monotonic()))
                self.probe("scheduled", planned)
            time.sleep(max(0, self.started + self.count * self.interval - time.monotonic()))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.ended = time.monotonic()

    def finish(self):
        self.declared_complete = time.monotonic()
        return {"declared_complete": True}

    def status(self):
        with self.records:
            return json.loads(json.dumps({"started_monotonic_s": self.started, "ended_monotonic_s": self.ended,
                "declared_complete_monotonic_s": self.declared_complete, "error": self.error,
                "events": self.events, "journal": self.journal}))


def serve(base_url, output):
    traffic = Traffic(base_url, output)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            actions = {"/start": traffic.start, "/finish": traffic.finish, "/status": traffic.status,
                       "/baseline": lambda: traffic.probe("baseline"), "/check": lambda: traffic.probe("diagnostic")}
            if self.path not in actions:
                self.send_error(404)
                return
            body = json.dumps(actions[self.path]()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", 8099), Handler).serve_forever()


if __name__ == "__main__":
    import sys

    serve(sys.argv[1], Path("/tmp/requests.jsonl"))
