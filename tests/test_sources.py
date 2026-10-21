import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from sources import (  # noqa: E402
    is_model_file,
    safe_filename,
    parse_github_repo,
    download_url,
    github_list_models,
)


def test_is_model_file():
    assert is_model_file("model.gguf")
    assert is_model_file("MODEL.GGUF")
    assert is_model_file("weights.safetensors")
    assert is_model_file("pytorch_model.bin")
    assert is_model_file("m.pt")
    assert is_model_file("m.pth")
    assert is_model_file("m.onnx")
    assert not is_model_file("readme.txt")
    assert not is_model_file("model.gguf.zip")


def test_safe_filename_strips_paths():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("/tmp/x.gguf") == "x.gguf"


def test_safe_filename_sanitizes():
    assert safe_filename("my model (1).gguf") == "my_model__1_.gguf"
    assert safe_filename("") == "model.bin"


def test_safe_filename_truncates():
    assert len(safe_filename("a" * 200 + ".gguf")) == 120


def test_parse_github_repo():
    assert parse_github_repo("https://github.com/user/repo") == ("user", "repo")
    assert parse_github_repo("https://github.com/user/repo.git") == ("user", "repo")
    assert parse_github_repo("https://github.com/user/repo/tree/main") == ("user", "repo")
    assert parse_github_repo("http://github.com/a/b/") == ("a", "b")


def test_parse_github_repo_rejects_bad():
    assert parse_github_repo("not a url") is None
    assert parse_github_repo("https://gitlab.com/user/repo") is None
    assert parse_github_repo("") is None


def test_download_url_rejects_non_http(tmp_path):
    dest = str(tmp_path / "out.bin")
    for bad in ("file:///etc/passwd", "ftp://example.com/x.gguf",
                "gopher://x/y", "x.gguf", ""):
        try:
            download_url(bad, dest)
        except ValueError as e:
            assert "http(s)" in str(e)
        else:
            raise AssertionError("expected ValueError for %r" % bad)


def test_github_list_models_bad_url():
    files, err = github_list_models("not a url")
    assert files is None
    assert isinstance(err, str) and err
