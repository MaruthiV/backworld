import http.client
import json
import sys
import time
import urllib.error
import urllib.request


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



if __name__ == "__main__":
    print(json.dumps(request(sys.argv[1], float(sys.argv[2]))))
