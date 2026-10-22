import hashlib
import http.client
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


def handler(base_url, record):
    target = urlsplit(base_url)
    if target.scheme != "http" or not target.hostname or target.username or target.password or target.path or target.query or target.fragment:
        raise ValueError("relay requires one fixed HTTP origin")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc or parsed.path not in ("/hotels", "/reservation"):
                self.send_error(404)
                return
            started = time.monotonic()
            request_id = hashlib.sha256(self.path.encode()).hexdigest()
            event = {"id": uuid.uuid4().hex, "request_hash": request_id, "path": parsed.path, "started_unix_s": time.time()}
            event["client_request_id"] = self.headers.get("X-Backworld-Request")
            record({"stage": "started", **event})
            connection = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=4)
            status, body, content_type, error = 502, b'{"error":"upstream transport failed"}', "application/json", None
            try:
                connection.request("GET", self.path)
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


def serve(base_url):
    lock = threading.Lock()

    def record(event):
        with lock:
            print(json.dumps(event), flush=True)

    class Server(ThreadingHTTPServer):
        request_queue_size = 256

    Server(("0.0.0.0", 8080), handler(base_url, record)).serve_forever()


if __name__ == "__main__":
    import sys

    serve(sys.argv[1])
