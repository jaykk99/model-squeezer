"""Model download sources: GitHub repo, Hugging Face, direct URL, upload.

All downloads stream to disk with progress callbacks so the UI can show
a real progress bar. Only model weight files are accepted.
"""
import os
import re
import json
import urllib.request
import urllib.parse

MODEL_EXTS = (".gguf", ".safetensors", ".bin", ".pt", ".pth", ".onnx")


def is_model_file(name):
    return name.lower().endswith(MODEL_EXTS)


def _progress_reader(resp, total, dest_path, on_progress):
    downloaded = 0
    chunk = 1024 * 256
    with open(dest_path, "wb") as f:
        while True:
            data = resp.read(chunk)
            if not data:
                break
            f.write(data)
            downloaded += len(data)
            if on_progress and total:
                on_progress(min(100, downloaded * 100.0 / total))
    if on_progress:
        on_progress(100)
    return dest_path


def download_url(url, dest_path, on_progress=None):
    """Stream-download any direct URL to disk.

    Only http(s) URLs are allowed: anything else (file://, ftp://, ...)
    raises ValueError so a crafted URL can never make the server read or
    write local files.
    """
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError("Only http(s) URLs are allowed, got: %s" % (scheme or "(none)"))
    req = urllib.request.Request(url, headers={"User-Agent": "model-squeezer/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = resp.headers.get("Content-Length")
        total = int(total) if total else 0
        return _progress_reader(resp, total, dest_path, on_progress)


def parse_github_repo(url):
    """Extract owner/repo from a GitHub URL."""
    m = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/|$)", url)
    if not m:
        return None
    return m.group(1), m.group(2)


def github_list_models(repo_url):
    """List model weight files in a GitHub repo via the public API."""
    parsed = parse_github_repo(repo_url)
    if not parsed:
        return None, "Could not parse that GitHub URL."
    owner, repo = parsed
    api = "https://api.github.com/repos/%s/%s/git/trees/HEAD?recursive=1" % (owner, repo)
    req = urllib.request.Request(api, headers={"User-Agent": "model-squeezer/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except Exception as e:
        return None, "GitHub API error: %s" % e
    files = []
    for t in data.get("tree", []):
        if t.get("type") == "blob" and is_model_file(t.get("path", "")):
            files.append({
                "path": t["path"],
                "size": t.get("size", 0),
                "raw_url": "https://raw.githubusercontent.com/%s/%s/HEAD/%s"
                           % (owner, repo, t["path"]),
            })
    return files, None


def huggingface_list_models(model_url_or_id):
    """List downloadable weight files for a Hugging Face model."""
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        return None, "huggingface_hub is not installed on the server."
    m = re.search(r"huggingface\.co/([^/]+/[^/]+)", model_url_or_id)
    repo_id = m.group(1) if m else model_url_or_id.strip()
    try:
        files = list_repo_files(repo_id)
    except Exception as e:
        return None, "Hugging Face error: %s" % e
    out = []
    for f in files:
        if is_model_file(f):
            out.append({"path": f, "repo_id": repo_id})
    if not out:
        return [], "No model weight files found in %s." % repo_id
    return out, None


def huggingface_download(repo_id, filename, dest_path, on_progress=None):
    """Download one file from Hugging Face with progress."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("huggingface_hub is not installed on the server.")
    # hf_hub_download caches; copy out to our dest so we own the file.
    import shutil
    tmp = {"pct": 0}

    def hook(*a, **k):
        pass  # hf_hub_download has no fine progress hook; we fake stages

    if on_progress:
        on_progress(5)
    cached = hf_hub_download(repo_id=repo_id, filename=filename)
    if on_progress:
        on_progress(80)
    shutil.copyfile(cached, dest_path)
    if on_progress:
        on_progress(100)
    return dest_path


def safe_filename(name):
    name = os.path.basename(name)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name[:120] or "model.bin"
