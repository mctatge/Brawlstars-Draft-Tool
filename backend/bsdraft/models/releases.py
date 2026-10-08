"""Versioned model releases and their small, atomically promoted manifest (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

SCHEMA_VERSION = 1
POINTER_TAG = "model-current"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported model manifest schema")
    for key in ("release_id", "published_at", "trained_at", "source_commit", "dataset_sha256"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"manifest missing {key}")
    if not _SHA.fullmatch(manifest["dataset_sha256"]):
        raise ValueError("invalid dataset digest")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"]):
        raise ValueError("invalid source commit")
    analysis = manifest.get("analysis")
    if not isinstance(analysis, dict) or not isinstance(analysis.get("era_id"), str):
        raise ValueError("manifest missing analysis era")
    if not isinstance(analysis.get("start_ts"), int) or analysis["start_ts"] <= 0:
        raise ValueError("manifest missing active analysis cutoff")
    for name in ("model", "metrics"):
        artifact = manifest.get(name)
        if not isinstance(artifact, dict) or not _SHA.fullmatch(str(artifact.get("sha256", ""))):
            raise ValueError(f"invalid {name} digest")
        parsed = urlparse(artifact.get("url", ""))
        if parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.username:
            raise ValueError(f"invalid {name} release URL")
        if "/releases/download/" not in parsed.path or parsed.query or parsed.fragment:
            raise ValueError(f"invalid {name} release URL")
        if not isinstance(artifact.get("size_bytes"), int) or artifact["size_bytes"] <= 0:
            raise ValueError(f"invalid {name} size")
    return manifest


def parse_manifest(payload) -> dict:
    """Read a plain manifest or the public GitHub release response containing it in body."""
    if isinstance(payload, (str, bytes)):
        payload = json.loads(payload)
    if isinstance(payload, dict) and "body" in payload:
        payload = json.loads(payload["body"])
    return validate_manifest(payload)


def download_verified(url: str, destination: Path, *, sha256: str = "", size_bytes: int = 0,
                      max_bytes: int = 8 * 1024 * 1024, deadline_seconds: float = 60.0) -> None:
    """Stage small public release files; failed downloads never replace the destination."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    try:
        deadline = time.monotonic() + deadline_seconds
        request = urllib.request.Request(url, headers={"User-Agent": "BrawlDraft-model-release"})
        with urllib.request.urlopen(request, timeout=min(30.0, deadline_seconds)) as response, temporary.open("wb") as out:
            total = 0
            # HTTPResponse.read1 returns currently available bytes instead of waiting for a
            # whole buffer from a trickle source. Check a total deadline between socket reads.
            read = getattr(response, "read1", response.read)
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError("model release download exceeded total deadline")
                chunk = read(64 * 1024)
                if time.monotonic() >= deadline:
                    raise TimeoutError("model release download exceeded total deadline")
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("model release download exceeds size limit")
                out.write(chunk)
        if size_bytes and total != size_bytes:
            raise ValueError("model release size mismatch")
        if sha256 and sha256_file(temporary) != sha256:
            raise ValueError("model release digest mismatch")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def download_incumbent(repository: str, destination: Path) -> dict | None:
    """Only a 404 of the new pointer permits the one-time legacy release migration."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid GitHub repository")
    pointer = f"https://api.github.com/repos/{repository}/releases/tags/{POINTER_TAG}"
    request = urllib.request.Request(pointer, headers={"User-Agent": "BrawlDraft-model-release"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read(64 * 1024 + 1)
        if len(payload) > 64 * 1024:
            raise ValueError("model pointer exceeds size limit")
        manifest = parse_manifest(payload)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        download_verified(f"https://github.com/{repository}/releases/download/data-latest/winprob.npz",
                          destination)
        return None
    download_verified(manifest["model"]["url"], destination,
                      sha256=manifest["model"]["sha256"],
                      size_bytes=manifest["model"]["size_bytes"])
    return manifest
