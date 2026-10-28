import hashlib
import http.client
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


def handler(base_url, record, profile="hotel"):
    target = urlsplit(base_url)
    if target.scheme != "http" or not target.hostname or target.username or target.password or target.path or target.query or target.fragment:
        raise ValueError("relay requires one fixed HTTP origin")
    if profile not in ("hotel", "shop"):
        raise ValueError("unknown relay profile")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.forward("GET")

        def do_POST(self):
            self.forward("POST")

        def forward(self, method):
            self.connection.settimeout(5)
            parsed = urlsplit(self.path)
            paths = ("/hotels", "/reservation") if profile == "hotel" else ("/api/products", "/api/cart")
            if parsed.scheme or parsed.netloc or parsed.fragment or parsed.path not in paths or (method == "POST" and (profile != "shop" or parsed.path != "/api/cart" or parsed.query)):
                self.send_error(404)
                return
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get("Transfer-Encoding") or len(lengths) > 1 or (method == "POST" and not lengths):
                self.send_error(400)
                return
            if lengths and (not lengths[0].isascii() or not lengths[0].isdecimal()):
                self.send_error(400)
                return
            size = int(lengths[0]) if lengths else 0
            if size > 4096 or (method == "GET" and size):
                self.send_error(413)
                return
            payload = self.rfile.read(size)
            if len(payload) != size:
                self.send_error(400)
                return
            started = time.monotonic()
            request_id = hashlib.sha256(self.path.encode()).hexdigest()
            event = {"id": uuid.uuid4().hex, "request_hash": request_id, "path": parsed.path, "started_unix_s": time.time()}
            event.update({"method": method, "request_body_sha256": hashlib.sha256(payload).hexdigest()})
            event["client_request_id"] = self.headers.get("X-Backworld-Request")
            record({"stage": "started", **event})
            connection = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=4)
            status, body, content_type, error = 502, b'{"error":"upstream transport failed"}', "application/json", None
            try:
                connection.request(method, self.path, body=payload if method == "POST" else None,
                                   headers={"Content-Type": "application/json"} if method == "POST" else {})
                response = connection.getresponse()
                status, body = response.status, response.read(65537)
                if len(body) > 65536:
                    raise ValueError("upstream body exceeds the limit")
                content_type = response.getheader("Content-Type", "application/json")
            except (OSError, ValueError, http.client.HTTPException) as exc:
                status, body, content_type, error = 502, b'{"error":"upstream transport failed"}', "application/json", type(exc).__name__
            finally:
                connection.close()
            delivered = False
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Backworld-Admission", event["id"])
                self.end_headers()
                self.wfile.write(body)
                delivered = True
            except OSError:
                pass
            finally:
                record({"stage": "completed", **event, "status": status, "bytes": len(body),
                        "body_sha256": hashlib.sha256(body).hexdigest(), "error": error,
                        "delivered": delivered, "elapsed_s": time.monotonic() - started})

        def log_message(self, *args):
            pass

    return Handler


def serve(base_url, profile="hotel"):
    lock = threading.Lock()

    def record(event):
        with lock:
            print(json.dumps(event), flush=True)

    class Server(ThreadingHTTPServer):
        request_queue_size = 256

    Server(("0.0.0.0", 8080), handler(base_url, record, profile)).serve_forever()


if __name__ == "__main__":
    import sys

    serve(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "hotel")
