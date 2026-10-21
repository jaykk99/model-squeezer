"""model-squeezer: compress AI models as many times as you want.

Tabs: Compress | Chromebook Mode | Train | History
Sources: upload, GitHub repo, Hugging Face, direct URL.
"""
import os
import uuid
import json
import time
import shutil
import threading
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    # Package mode (uvicorn app.main:app, the Docker/Render entrypoint)
    from app.quant import (QUANT_LEVELS, QUANT_TYPE_TO_LEVEL, BITS_PER_PARAM,
                           estimate_size_mb, fmt_mb, gguf_param_count,
                           run_quantize, chromebook_plan, CHROMEBOOK_RAM_MB,
                           find_quantize_bin, normalize_level, size_estimates_mb)
    from app.sources import (download_url, github_list_models, huggingface_list_models,
                             huggingface_download, safe_filename, parse_github_repo)
except ImportError:
    # Script mode (uvicorn main:app from inside app/)
    from quant import (QUANT_LEVELS, QUANT_TYPE_TO_LEVEL, BITS_PER_PARAM,
                       estimate_size_mb, fmt_mb, gguf_param_count,
                       run_quantize, chromebook_plan, CHROMEBOOK_RAM_MB,
                       find_quantize_bin, normalize_level, size_estimates_mb)
    from sources import (download_url, github_list_models, huggingface_list_models,
                         huggingface_download, safe_filename, parse_github_repo)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SQ_DATA", os.path.join(BASE, "..", "data"))
MODELS_DIR = os.path.join(DATA, "models")
OUT_DIR = os.path.join(DATA, "outputs")
TRAIN_DIR = os.path.join(DATA, "trained")
for d in (MODELS_DIR, OUT_DIR, TRAIN_DIR):
    os.makedirs(d, exist_ok=True)

REG_PATH = os.path.join(DATA, "models.json")
HIST_PATH = os.path.join(DATA, "history.json")

app = FastAPI(title="model-squeezer")

jobs = {}
job_lock = threading.Lock()


def _load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def registry():
    return _load_json(REG_PATH, {})


def save_registry(reg):
    _save_json(REG_PATH, reg)


def history():
    return _load_json(HIST_PATH, [])


def add_history(entry):
    h = history()
    h.insert(0, entry)
    _save_json(HIST_PATH, h[:200])


def new_job(kind):
    jid = uuid.uuid4().hex[:12]
    with job_lock:
        jobs[jid] = {"id": jid, "kind": kind, "progress": 0,
                     "status_msg": "queued", "ok": None, "error": None,
                     "created": datetime.utcnow().isoformat()}
    return jid


def set_job(jid, **kw):
    with job_lock:
        if jid in jobs:
            jobs[jid].update(kw)


def mb_of(path):
    return os.path.getsize(path) / 1024 / 1024


# ---------------- models ----------------

class FetchReq(BaseModel):
    kind: str          # github | huggingface | url
    url: str
    filename: str = ""  # picked file for github/hf; required for url-derived name
    repo_id: str = ""   # for huggingface


def _register_model(name, path, source, origin_id=None):
    reg = registry()
    mid = uuid.uuid4().hex[:12]
    size = mb_of(path)
    params = gguf_param_count(path) if name.lower().endswith(".gguf") else None
    reg[mid] = {"id": mid, "name": name, "path": path,
                "size_mb": round(size, 1), "source": source,
                "params": params,
                "params_b": round(params / 1e9, 2) if params else None,
                "created": datetime.utcnow().isoformat(),
                "origin_id": origin_id}
    save_registry(reg)
    return reg[mid]


def _do_fetch(jid, kind, url, filename, repo_id):
    try:
        dest_name = safe_filename(filename or url.split("?")[0].rsplit("/", 1)[-1]
                                  or "model.bin")
        dest = os.path.join(MODELS_DIR, "%s_%s" % (jid, dest_name))

        def prog(p):
            set_job(jid, progress=p, status_msg="Downloading... %d%%" % p)

        set_job(jid, progress=1, status_msg="Downloading...")
        if kind == "url":
            download_url(url, dest, on_progress=prog)
            source = "url"
        elif kind == "github":
            # filename holds the raw_url here
            download_url(filename, dest, on_progress=prog)
            source = "github"
        elif kind == "huggingface":
            huggingface_download(repo_id, filename, dest, on_progress=prog)
            source = "huggingface"
        else:
            raise ValueError("unknown source kind")

        if os.path.getsize(dest) == 0:
            raise ValueError("downloaded file is empty")
        model = _register_model(dest_name, dest, source)
        set_job(jid, ok=True, progress=100, status_msg="Ready.",
                result={"model": model})
    except Exception as e:
        set_job(jid, ok=False, progress=100,
                status_msg="Download failed.", error=str(e)[:500])


@app.post("/api/fetch")
def api_fetch(req: FetchReq, bg: BackgroundTasks):
    if req.kind not in ("github", "huggingface", "url"):
        raise HTTPException(400, "bad kind")
    if req.kind in ("github", "huggingface") and not req.filename:
        raise HTTPException(400, "pick a file first")
    jid = new_job("fetch")
    t = threading.Thread(target=_do_fetch,
                         args=(jid, req.kind, req.url, req.filename, req.repo_id),
                         daemon=True)
    t.start()
    return {"job_id": jid}


@app.get("/api/fetch/list")
def api_fetch_list(kind: str, url: str):
    if kind == "github":
        files, err = github_list_models(url)
        if err:
            raise HTTPException(400, err)
        return {"files": files}
    if kind == "huggingface":
        files, err = huggingface_list_models(url)
        if err:
            raise HTTPException(400, err)
        return {"files": files}
    raise HTTPException(400, "bad kind")


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    jid = new_job("fetch")
    # Read here: the request file is closed after we return, so the
    # background thread cannot read it lazily.
    data = await file.read()
    fname = file.filename or "model.bin"

    def _save():
        try:
            dest_name = safe_filename(fname)
            dest = os.path.join(MODELS_DIR, "%s_%s" % (jid, dest_name))
            set_job(jid, progress=50, status_msg="Saving...")
            with open(dest, "wb") as f:
                f.write(data)
            set_job(jid, progress=90, status_msg="Registering...")
            model = _register_model(dest_name, dest, "upload")
            set_job(jid, ok=True, progress=100, status_msg="Ready.",
                    result={"model": model})
        except Exception as e:
            set_job(jid, ok=False, progress=100,
                    status_msg="Upload failed.", error=str(e)[:500])

    t = threading.Thread(target=_save, daemon=True)
    t.start()
    return {"job_id": jid}


@app.get("/api/models")
def api_models():
    reg = registry()
    return {"models": sorted(reg.values(),
                             key=lambda m: m["created"], reverse=True)}


# ---------------- compression ----------------

class CompressReq(BaseModel):
    model_id: str
    level: str  # Q8, Q6, Q5, Q4, Q3, Q2, Q1


def _do_compress(jid, model_id, level):
    try:
        level = normalize_level(level)
        if not level:
            raise ValueError(
                "unknown level: pick one of %s (or a raw type like Q4_K_M)"
                % ", ".join(QUANT_LEVELS))
        reg = registry()
        m = reg.get(model_id)
        if not m:
            raise ValueError("model not found")
        # Always quantize from the ORIGINAL file. llama.cpp cannot
        # re-quantize an already quantized model, and going straight from
        # the original gives better quality anyway.
        origin = reg.get(m.get("origin_id") or model_id, m)
        in_path = origin["path"]
        in_name = origin["name"]
        if not in_path.lower().endswith(".gguf"):
            raise ValueError(
                "Only GGUF files can be quantized right now. "
                "Convert to GGUF first, or pick a GGUF model.")
        qtype = QUANT_LEVELS[level]
        base = os.path.splitext(in_name)[0]
        out_name = "%s-%s.gguf" % (base, qtype.replace("_", ""))
        out_path = os.path.join(OUT_DIR, "%s_%s" % (jid, out_name))

        set_job(jid, progress=5, status_msg="Quantizing to %s..." % qtype)

        def on_log(log):
            set_job(jid, log_tail=log)

        ok, msg = run_quantize(in_path, out_path, qtype, on_log=on_log)
        if not ok:
            raise RuntimeError(msg)

        before_mb = mb_of(in_path)
        after_mb = mb_of(out_path)
        saved = before_mb - after_mb
        pct = (saved / before_mb * 100) if before_mb else 0

        fid = uuid.uuid4().hex[:12]
        reg[fid] = {"id": fid, "name": out_name, "path": out_path,
                    "size_mb": round(after_mb, 1), "source": "compressed",
                    "params": origin.get("params"),
                    "params_b": origin.get("params_b"),
                    "created": datetime.utcnow().isoformat(),
                    "origin_id": origin["id"],
                    "from": model_id, "level": level, "quant_type": qtype}
        save_registry(reg)

        add_history({
            "id": jid, "at": datetime.utcnow().isoformat(),
            "input": in_name + (" (via " + m["name"] + ")" if m["id"] != origin["id"] else ""),
            "output": out_name,
            "level": level, "quant_type": qtype,
            "before_mb": round(before_mb, 1), "after_mb": round(after_mb, 1),
            "saved_mb": round(saved, 1), "saved_pct": round(pct, 1),
            "file_id": fid,
        })
        set_job(jid, ok=True, progress=100,
                status_msg="Done. %s -> %s" % (fmt_mb(before_mb), fmt_mb(after_mb)),
                result={"file_id": fid, "name": out_name,
                        "before_mb": round(before_mb, 1),
                        "after_mb": round(after_mb, 1),
                        "saved_mb": round(saved, 1),
                        "saved_pct": round(pct, 1),
                        "level": level})
    except Exception as e:
        set_job(jid, ok=False, progress=100,
                status_msg="Compression failed.", error=str(e)[:600])


@app.post("/api/compress")
def api_compress(req: CompressReq):
    jid = new_job("compress")
    t = threading.Thread(target=_do_compress,
                         args=(jid, req.model_id, req.level), daemon=True)
    t.start()
    return {"job_id": jid}


@app.get("/api/history")
def api_history(limit: int = 200):
    h = history()
    return {"history": h[:max(1, min(limit, 200))]}


@app.get("/api/levels")
def api_levels():
    return {"levels": [
        {"name": k, "quant_type": v,
         "bits": BITS_PER_PARAM[v]} for k, v in QUANT_LEVELS.items()]}


@app.get("/api/stats")
def api_stats():
    """Totals for the dashboard header: models, disk used, history, jobs."""
    reg = registry()
    total_mb = sum(m.get("size_mb", 0) or 0 for m in reg.values())
    with job_lock:
        active = sum(1 for j in jobs.values() if j.get("ok") is None)
    return {
        "models": len(reg),
        "total_size_mb": round(total_mb, 1),
        "history_entries": len(history()),
        "quantize_available": bool(find_quantize_bin()),
        "active_jobs": active,
    }


@app.get("/api/jobs")
def api_jobs(limit: int = 20):
    """Recent jobs, newest first. Long log tails are stripped here;
    fetch /api/jobs/{jid} for a single job's full detail."""
    limit = max(1, min(limit, 100))
    with job_lock:
        js = sorted(jobs.values(),
                    key=lambda j: j.get("created", ""), reverse=True)[:limit]
        out = []
        for j in js:
            j2 = dict(j)
            j2.pop("log_tail", None)
            out.append(j2)
    return {"jobs": out}


@app.get("/api/models/{mid}")
def api_model_get(mid: str):
    reg = registry()
    m = reg.get(mid)
    if not m:
        raise HTTPException(404, "model not found")
    return m


@app.get("/api/models/{mid}/estimates")
def api_model_estimates(mid: str):
    """Per-level size estimates for one model (needs known param count)."""
    reg = registry()
    m = reg.get(mid)
    if not m:
        raise HTTPException(404, "model not found")
    params = m.get("params")
    estimates = size_estimates_mb(params) if params else []
    return {
        "model_id": mid,
        "name": m["name"],
        "size_mb": m.get("size_mb"),
        "params_b": m.get("params_b"),
        "estimates": estimates,
        "note": (None if params else
                 "Param count unknown (not a GGUF with readable metadata); "
                 "per-level estimates need it."),
    }


@app.get("/api/models/{mid}/chain")
def api_model_chain(mid: str):
    """The compress-again chain this model belongs to, oldest first.

    Walks `from` links (each squeeze records the model it was squeezed
    from), so a Q8 -> Q4 -> Q2 chain shows as three steps.
    """
    reg = registry()
    if mid not in reg:
        raise HTTPException(404, "model not found")
    chain = []
    seen = set()
    cur = mid
    while cur and cur not in seen:
        seen.add(cur)
        m = reg.get(cur)
        if not m:
            break
        chain.append({
            "id": m["id"], "name": m["name"], "size_mb": m.get("size_mb"),
            "source": m.get("source"), "level": m.get("level"),
            "quant_type": m.get("quant_type"),
            "saved_pct": None,
        })
        nxt = m.get("from")
        if not nxt:
            oid = m.get("origin_id")
            nxt = oid if oid and oid != cur else None
        cur = nxt
    chain.reverse()
    return {"chain": chain}


@app.patch("/api/models/{mid}")
def api_model_rename(mid: str, body: dict):
    """Rename a model (only the display name; the file is untouched)."""
    reg = registry()
    m = reg.get(mid)
    if not m:
        raise HTTPException(404, "model not found")
    name = (body or {}).get("name", "")
    if not isinstance(name, str) or not name.strip():
        raise HTTPException(422, "name must be a non-empty string")
    m["name"] = name.strip()[:120]
    save_registry(reg)
    return {"ok": True, "id": mid, "name": m["name"]}


@app.get("/api/compress/preview")
def api_compress_preview(model_id: str, level: str):
    """Honest before/after prediction for one squeeze.

    The after-size is estimated from the model's param count, so it only
    works when the param count is known (GGUF with readable metadata).
    """
    level = normalize_level(level)
    if not level:
        raise HTTPException(400, "unknown level")
    reg = registry()
    m = reg.get(model_id)
    if not m:
        raise HTTPException(404, "model not found")
    qtype = QUANT_LEVELS[level]
    params = m.get("params")
    before_mb = m.get("size_mb")
    est_after = estimate_size_mb(params, qtype) if params else None
    saved_pct = None
    if before_mb and est_after:
        saved_pct = round((before_mb - est_after) / before_mb * 100, 1)
    return {
        "model_id": model_id, "name": m["name"], "level": level,
        "quant_type": qtype, "before_mb": before_mb,
        "est_after_mb": round(est_after, 1) if est_after else None,
        "est_saved_pct": saved_pct,
        "quantize_available": bool(find_quantize_bin()),
        "note": (None if params else
                 "Param count unknown (not a GGUF with readable metadata); "
                 "prediction needs it."),
    }


@app.get("/api/history/export")
def api_history_export(fmt: str = "json"):
    """Download the compression history as JSON or CSV."""
    h = history()
    if fmt == "csv":
        import csv
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["at", "input", "output", "level", "quant_type",
                    "before_mb", "after_mb", "saved_mb", "saved_pct"])
        for e in h:
            w.writerow([e.get("at"), e.get("input"), e.get("output"),
                        e.get("level"), e.get("quant_type"),
                        e.get("before_mb"), e.get("after_mb"),
                        e.get("saved_mb"), e.get("saved_pct")])
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                                 headers={"Content-Disposition":
                                          "attachment; filename=squeezer-history.csv"})
    return {"history": h}


@app.delete("/api/models/{mid}")
def api_model_delete(mid: str):
    """Delete a model and its file. Refuses when other registry entries
    were derived from it (compressed outputs, adapters) so chains of
    "compress again" runs never lose their source silently."""
    reg = registry()
    m = reg.get(mid)
    if not m:
        raise HTTPException(404, "model not found")
    children = [c["id"] for c in reg.values()
                if c.get("origin_id") == mid or c.get("from") == mid]
    if children:
        raise HTTPException(
            409, "Model is the source for %d other %s; delete those first."
            % (len(children), "entry" if len(children) == 1 else "entries"))
    try:
        if m.get("path") and os.path.isfile(m["path"]):
            os.remove(m["path"])
    except OSError:
        pass
    del reg[mid]
    save_registry(reg)
    return {"ok": True, "deleted": mid}


@app.delete("/api/history")
def api_history_clear():
    _save_json(HIST_PATH, [])
    return {"ok": True}


# ---------------- chromebook mode ----------------

@app.get("/api/chromebook/plan")
def api_chromebook_plan(params_b: float = None, model_id: str = None):
    pb = params_b
    if model_id:
        m = registry().get(model_id)
        if not m:
            raise HTTPException(404, "model not found")
        pb = m.get("params_b")
    if not pb:
        raise HTTPException(400, "need params_b or a model with known params")
    plan, verdict = chromebook_plan(int(pb * 1e9))
    return {"params_b": pb, "target_ram_mb": CHROMEBOOK_RAM_MB,
            "plan": plan, "verdict": verdict,
            "mmap_note": "llama.cpp memory-maps weights by default, so the OS "
                         "pages them in and out. Peak active RAM is still "
                         "roughly the model size plus context, so the fit "
                         "table above is what matters."}


# ---------------- training ----------------

@app.post("/api/train")
async def api_train(model_id: str = Form(...),
                    method: str = Form("lora"),
                    epochs: int = Form(1),
                    lr: float = Form(2e-4),
                    batch_size: int = Form(1),
                    grad_accum: int = Form(8),
                    max_seq_len: int = Form(256),
                    data: UploadFile = File(...)):
    reg = registry()
    m = reg.get(model_id)
    if not m:
        raise HTTPException(404, "model not found")
    mp = m["path"]
    if method not in ("lora", "lora-lowmem"):
        raise HTTPException(422, "method must be lora or lora-lowmem")
    if not (1 <= epochs <= 20):
        raise HTTPException(422, "epochs must be 1..20")
    if not (0 < lr <= 1):
        raise HTTPException(422, "lr must be positive and at most 1")
    if not (1 <= batch_size <= 8):
        raise HTTPException(422, "batch_size must be 1..8")
    if not (1 <= grad_accum <= 64):
        raise HTTPException(422, "grad_accum must be 1..64")
    if not (16 <= max_seq_len <= 4096):
        raise HTTPException(422, "max_seq_len must be 16..4096")
    jid = new_job("train")
    # Read now: request files close after we return.
    data_bytes = await data.read()
    if len(data_bytes) == 0:
        raise HTTPException(422, "training data file is empty")
    if len(data_bytes) > 200 * 1024 * 1024:
        raise HTTPException(413, "training data over 200 MB is not accepted")
    data_name = safe_filename(data.filename or "data.txt")
    data_path = os.path.join(TRAIN_DIR, "%s_%s" % (jid, data_name))
    with open(data_path, "wb") as f:
        f.write(data_bytes)
    out_dir = os.path.join(TRAIN_DIR, "%s_adapter" % jid)

    def _run():
        try:
            try:
                from app.train import run_lora_training  # package mode
            except ImportError:
                from train import run_lora_training  # script mode
            job = jobs[jid]
            run_lora_training(job, mp, data_path, out_dir,
                              method=method, epochs=epochs, lr=lr,
                              batch_size=batch_size, grad_accum=grad_accum,
                              max_seq_len=max_seq_len)
            if job.get("ok"):
                fid = uuid.uuid4().hex[:12]
                # zip the adapter for download
                zip_path = out_dir + ".zip"
                shutil.make_archive(out_dir, "zip", out_dir)
                reg2 = registry()
                reg2[fid] = {"id": fid,
                             "name": "lora-adapter-%s.zip" % jid[:6],
                             "path": zip_path,
                             "size_mb": round(mb_of(zip_path), 1),
                             "source": "trained",
                             "created": datetime.utcnow().isoformat()}
                save_registry(reg2)
                job["result"]["file_id"] = fid
        except Exception as e:
            set_job(jid, ok=False, progress=100,
                    status_msg="Training failed.", error=str(e)[:600])

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"job_id": jid}


@app.post("/api/train/cancel/{jid}")
def api_train_cancel(jid: str):
    with job_lock:
        if jid in jobs:
            jobs[jid]["cancel"] = True
    return {"ok": True}


# ---------------- jobs & downloads ----------------

@app.get("/api/jobs/{jid}")
def api_job(jid: str):
    with job_lock:
        j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "job not found")
    return j


@app.get("/api/download/{fid}")
def api_download(fid: str):
    reg = registry()
    m = reg.get(fid)
    if not m or not os.path.isfile(m["path"]):
        raise HTTPException(404, "file not found")
    return FileResponse(m["path"], filename=m["name"])


@app.get("/api/health")
def api_health():
    return {"ok": True, "quantize": bool(find_quantize_bin()),
            "time": int(time.time())}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(BASE, "static", "index.html")) as f:
        return f.read()
