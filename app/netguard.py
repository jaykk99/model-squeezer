"""Bounded, public-network-only downloads for user-supplied model URLs.

Rejects private/reserved addresses and revalidates redirects. DNS can change
between resolution and connection, so this is defense in depth, not a complete
DNS-rebinding defense; production should also block private egress at the
network layer.
"""
from __future__ import annotations

import ipaddress
import os
import socket
import urllib.parse
import urllib.request

DEFAULT_MAX_BYTES = 10 * 1024 * 1024 * 1024  # configurable 10 GiB ceiling


def require_public_url(url: str) -> str:
    if not isinstance(url, str):
        raise ValueError("Only http(s) URLs are allowed")
    try:
        parsed = urllib.parse.urlsplit(url)
        scheme = parsed.scheme.lower()
        if scheme not in ("http", "https"):
            raise ValueError("Only http(s) URLs are allowed")
        if not parsed.hostname or parsed.username is not None or parsed.password is not None:
            raise ValueError("A public hostname without embedded credentials is required")
        port = parsed.port or (443 if scheme == "https" else 80)
        addrs = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except (ValueError, socket.gaierror, UnicodeError) as exc:
        raise ValueError("Invalid or unresolved model URL: %s" % exc) from exc
    if not addrs:
        raise ValueError("Model hostname did not resolve")
    for addr in addrs:
        try:
            ip = ipaddress.ip_address(addr[4][0])
        except ValueError as exc:
            raise ValueError("Unrecognized model host address") from exc
        if not ip.is_global:
            raise ValueError("Model URL points to a non-public network address")
    return url


class PublicRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, new_url):
        require_public_url(new_url)
        return super().redirect_request(request, fp, code, msg, headers, new_url)


def download_public(url: str, destination: str, on_progress=None, max_bytes=None) -> str:
    """Download with a bounded body and reject internal hosts and redirects.

    On download errors, partial destination files are removed. Limits apply
    to actual downloaded bytes even when Content-Length is missing or wrong.
    """
    require_public_url(url)
    if max_bytes is None:
        max_bytes = os.environ.get("SQ_MAX_DOWNLOAD_BYTES", str(DEFAULT_MAX_BYTES))
    max_bytes = int(max_bytes)
    if max_bytes < 1:
        raise ValueError("Download byte ceiling must be positive")
    req = urllib.request.Request(url, headers={"User-Agent": "model-squeezer/1.0"})
    opener = urllib.request.build_opener(PublicRedirects())
    started = False
    try:
        with opener.open(req, timeout=60) as resp:
            size = resp.headers.get("Content-Length")
            size = int(size) if size is not None else 0
            if size > max_bytes:
                raise ValueError("Model download exceeds the configured size limit")
            count = 0
            with open(destination, "wb") as out:
                started = True
                while True:
                    chunk = resp.read(min(256 * 1024, max_bytes - count + 1))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > max_bytes:
                        raise ValueError("Model download exceeds the configured size limit")
                    out.write(chunk)
                    if on_progress and size > 0:
                        on_progress(min(100, count * 100.0 / size))
        if on_progress:
            on_progress(100)
        return destination
    except Exception:
        if started:
            try:
                os.remove(destination)
            except FileNotFoundError:
                pass
        raise
