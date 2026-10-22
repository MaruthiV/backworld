import argparse
import hashlib
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from service_operator import hotel


ORIGINAL_BINARY_SHA256 = "18628ff03c7f47cc1da3ba9262f40b673719f9b5e4fe573ffb9c88a6c3d54f76"
ORIGINAL_SOURCE_SHA256 = "22cb2469a00a8653cb8794edfbc684da8709f54e035590e2fc1e320e7744d458"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    base = hotel.IMAGES["app"]["ref"]
    rebuild = json.loads((inputs / "rebuild.json").read_text())
    if not rebuild["bit_identical"] or any(digest(inputs / name) != ORIGINAL_BINARY_SHA256
                                          for name in ("reservation.original", "reservation.rebuilt")):
        raise ValueError("unmodified rebuild does not match the published binary")
    manifest = json.loads((inputs / "source-manifest.json").read_text())
    for entry in manifest:
        if digest(inputs / "image-project" / entry["path"]) != entry["sha256"]:
            raise ValueError("extracted source changed: " + entry["path"])
    if digest(inputs / "image-project/services/reservation/server.go") != ORIGINAL_SOURCE_SHA256:
        raise ValueError("reservation source differs from the audited original")
    source = output / "source"
    shutil.copytree(inputs / "image-project", source)
    patch = Path("patches/hotel-reservation-availability.patch").resolve()
    (output / "availability.patch").write_bytes(patch.read_bytes())
    (output / "build_hotel_availability.py").write_bytes(Path(__file__).read_bytes())
    docker = ["docker", "--context", "desktop-linux"]
    commands = []

    def command(args, timeout=30, cwd=None, check=True):
        clock = time.monotonic()
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        commands.append({"args": args, "returncode": result.returncode, "elapsed_s": time.monotonic() - clock})
        (output / f"command-{len(commands):02d}.log").write_text(result.stdout + result.stderr)
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if check and result.returncode:
            raise RuntimeError("command failed; see " + str(output / f"command-{len(commands):02d}.log"))
        return result.stdout

    command(["patch", "-p1", "--batch", "--forward", "-i", str(patch)], cwd=source)
    changed = [entry["path"] for entry in manifest if digest(source / entry["path"]) != entry["sha256"]]
    if changed != ["services/reservation/server.go"]:
        raise ValueError("patch changed more than the declared reservation source")
    name = "bw-availability-build-" + uuid.uuid4().hex[:10]
    context = output / "context"
    context.mkdir()
    try:
        command(docker + ["run", "--rm", "--platform", "linux/amd64", "--name", name,
                "--label", "backworld.build=" + name, "--network", "none", "--memory", "2g", "--memory-swap", "2g",
                "--cpus", "2", "--pids-limit", "256", "--read-only", "--tmpfs", "/tmp:rw,size=768m",
                "--mount", "type=bind,src=" + str(source) + ",dst=/go/src/github.com/harlow/go-micro-services,readonly",
                "--mount", "type=bind,src=" + str(context) + ",dst=/out",
                "--workdir", "/go/src/github.com/harlow/go-micro-services", "-e", "GOCACHE=/tmp/go-cache",
                "-e", "GOPROXY=off", "-e", "GOSUMDB=off", "-e", "GOFLAGS=-mod=vendor", base,
                "go", "build", "-ldflags=-s -w", "-o", "/out/reservation", "./cmd/reservation"], timeout=240)
    finally:
        command(docker + ["rm", "-f", name], check=False)
    receipt = {"base_image": base, "original_binary_sha256": ORIGINAL_BINARY_SHA256,
               "unmodified_binary_matches": True, "patch_sha256": digest(patch),
               "corrected_binary_sha256": digest(context / "reservation"),
               "original_source_sha256": ORIGINAL_SOURCE_SHA256,
               "corrected_source_sha256": digest(source / "services/reservation/server.go"),
               "source_manifest_sha256": digest(inputs / "source-manifest.json"),
               "compiler": "Go 1.17.3 from pinned original image; offline vendored build",
               "scope": "only reservation binary/source replaced; not an agent repair or official benchmark image"}
    (context / "availability-build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    shutil.copyfile(source / "services/reservation/server.go", context / "server.go")
    (context / "Dockerfile").write_text("\n".join([
        "FROM " + base,
        "COPY reservation /go/bin/reservation",
        "COPY server.go /go/src/github.com/harlow/go-micro-services/services/reservation/server.go",
        "COPY availability-build.json /opt/backworld/availability-build.json", ""]))
    tag = "backworld.local/hotel-availability:" + receipt["patch_sha256"][:12]
    command(docker + ["build", "--platform", "linux/amd64", "--network", "none", "--pull=false", "--provenance=false",
                     "-t", tag, str(context)], timeout=180)
    info = json.loads(command(docker + ["image", "inspect", tag]))[0]
    if info["Architecture"] != "amd64":
        raise ValueError("corrected image architecture differs")
    receipt.update({"image_id": info["Id"], "observed_tag": tag, "elapsed_s": time.monotonic() - started})
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=Path("results/service-availability-source-20261022"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.inputs.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
