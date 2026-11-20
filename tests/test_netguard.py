"""Network-blocking and transfer-limit checks; no real outbound requests."""
import io
import socket
import urllib.request
from unittest.mock import patch

import pytest

from app.netguard import PublicRedirects, download_public, require_public_url


def public_dns(*args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 443))]


def private_dns(*args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7", 443))]


def test_local_and_reserved_ips_blocked_even_without_dns():
    for url in ["http://127.0.0.1/a", "http://169.254.169.254/latest/meta-data",
                "http://[::1]/", "http://192.168.0.9/x", "http://10.5.5.5/x"]:
        with pytest.raises(ValueError, match="non-public"):
            require_public_url(url)


def test_hostname_with_private_dns_blocked():
    with patch("socket.getaddrinfo", side_effect=private_dns):
        with pytest.raises(ValueError, match="non-public"):
            require_public_url("https://private.example/model.gguf")


def test_public_https_permitted_and_file_url_rejected():
    with patch("socket.getaddrinfo", side_effect=public_dns):
        assert require_public_url("https://example.org/model.gguf")
    with pytest.raises(ValueError, match=r"http\(s\)"):
        require_public_url("file:///etc/passwd")


def test_embedded_credentials_rejected():
    with pytest.raises(ValueError):
        require_public_url("http://username:secret@example.org/model.bin")


def test_redirect_to_private_ip_rejected():
    with pytest.raises(ValueError, match="non-public"):
        PublicRedirects().redirect_request(
            urllib.request.Request("https://public.example/model.gguf"),
            None, 302, "found", {}, "http://127.0.0.1/secrets")


class MockResponse:
    def __init__(self, data, content_length=None):
        self.stream = io.BytesIO(data)
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}

    def read(self, n=-1):
        return self.stream.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_streaming_limit_and_partial_file_cleanup(tmp_path):
    dest = tmp_path / "weights.gguf"
    with patch("socket.getaddrinfo", side_effect=public_dns), patch("urllib.request.build_opener") as opener:
        opener.return_value.open.return_value = MockResponse(b"0123456789")
        with pytest.raises(ValueError, match="size limit"):
            download_public("https://public.example/file.gguf", str(dest), max_bytes=4)
    assert not dest.exists()


def test_content_length_limit_blocks_before_file_write(tmp_path):
    dest = tmp_path / "weights.gguf"
    with patch("socket.getaddrinfo", side_effect=public_dns), patch("urllib.request.build_opener") as opener:
        opener.return_value.open.return_value = MockResponse(b"tiny", content_length=999)
        with pytest.raises(ValueError, match="size limit"):
            download_public("https://public.example/file.gguf", str(dest), max_bytes=4)
    assert not dest.exists()


def test_success_streaming_and_progress(tmp_path):
    dest = tmp_path / "weights.gguf"
    seen = []
    with patch("socket.getaddrinfo", side_effect=public_dns), patch("urllib.request.build_opener") as opener:
        opener.return_value.open.return_value = MockResponse(b"abcdefghij", content_length=10)
        assert download_public("https://public.example/file.gguf", str(dest), seen.append, max_bytes=11) == str(dest)
    assert dest.read_bytes() == b"abcdefghij"
    assert seen[-1] == 100
