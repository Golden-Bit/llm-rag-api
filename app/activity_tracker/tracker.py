# app/activity_tracker/tracker.py
from __future__ import annotations
import os, json, uuid, re, time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

# Cartella radice per i log attività (configurabile via env)
ACTIVITY_ROOT = Path(os.getenv("ACTIVITY_ROOT", "activity_logs"))
ACTIVITY_ROOT.mkdir(parents=True, exist_ok=True)

# Limite di bytes da conservare come preview della risposta stream
MAX_PREVIEW_BYTES = int(os.getenv("ACTIVITY_MAX_PREVIEW_BYTES", "50000"))

ActivityType = Literal["UPLOAD_ASYNC", "STREAM_EVENTS"]
ActivityStatus = Literal["PENDING", "RUNNING", "COMPLETED", "ERROR"]

SENSITIVE_KEYS = re.compile(r"(api[_-]?key|access[_-]?token|authorization|password)", re.I)

# ---------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat()

# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------
def _user_dir(user_id: str) -> Path:
    p = ACTIVITY_ROOT / user_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def _activity_path(user_id: str, activity_id: str) -> Path:
    return _user_dir(user_id) / f"{activity_id}.json"

# ---------------------------------------------------------------------
# I/O robusto (Windows-safe, UTF-8 sempre)
# ---------------------------------------------------------------------
def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
    """
    Scrive JSON UTF-8 in modo atomico:
      1) scrive su <file>.tmp con encoding='utf-8'
      2) flush + fsync
      3) os.replace(tmp, path)
    Così evitiamo file parziali e mismatch di encoding (es. Windows cp1252).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(data, ensure_ascii=False, indent=2)
    # scrittura esplicita in UTF-8
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    # sostituzione atomica sulla stessa partizione
    os.replace(tmp, path)

def _try_read_json_with_encoding(path: Path, encoding: str) -> Dict[str, Any]:
    with path.open("r", encoding=encoding, errors="strict") as f:
        return json.loads(f.read())

def _safe_load_json(path: Path, *, retries: int = 3, delay: float = 0.02) -> Dict[str, Any]:
    """
    Lettura resiliente:
      - tenta UTF-8 (standard)
      - fallback UTF-8-SIG
      - fallback latin-1
      - ultimo tentativo: decodifica 'utf-8' con errors='ignore'
    In caso di JSON corrotto per race, riprova qualche millisecondo dopo.
    """
    for attempt in range(retries):
        try:
            return _try_read_json_with_encoding(path, "utf-8")
        except UnicodeDecodeError:
            # fallback BOM
            try:
                return _try_read_json_with_encoding(path, "utf-8-sig")
            except UnicodeDecodeError:
                # fallback latin-1 (per vecchi file scritti con cp1252/latin-1)
                try:
                    return _try_read_json_with_encoding(path, "latin-1")
                except UnicodeDecodeError:
                    pass
        except json.JSONDecodeError:
            # file appena sostituito? attendo e riprovo
            time.sleep(delay)
            continue

        # Ultimo disperato tentativo: ignora caratteri non decodificabili
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="ignore")
            return json.loads(text)
        except Exception:
            # attendo e ritento se rimangono tentativi
            time.sleep(delay)

    # se siamo qui, tutti i tentativi sono falliti
    raise

def _load(user_id: str, activity_id: str) -> Dict[str, Any]:
    p = _activity_path(user_id, activity_id)
    if not p.exists():
        raise FileNotFoundError(f"Activity {activity_id} not found for {user_id}")
    return _safe_load_json(p)

# ---------------------------------------------------------------------
# Serializzazione/scrubbing
# ---------------------------------------------------------------------
def _safe_serialize(obj: Any) -> Any:
    # garantisce serializzabilità JSON (conversione bytes → breve stringa)
    if isinstance(obj, (bytes, bytearray)):
        return obj[:512].decode("utf-8", errors="ignore")
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return obj
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return str(obj)

def scrub_payload(data: Any) -> Any:
    """Rimuove o maschera chiavi sensibili ricorsivamente."""
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if SENSITIVE_KEYS.search(k):
                out[k] = "<redacted>"
            else:
                out[k] = scrub_payload(v)
        return out
    if isinstance(data, list):
        return [scrub_payload(x) for x in data]
    return _safe_serialize(data)

# ---------------------------------------------------------------------
# API principali
# ---------------------------------------------------------------------
def create_activity(
    *,
    user_id: str,
    activity_type: ActivityType,
    cost_usd: Optional[float] = None,
    payload: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    activity_id = str(uuid.uuid4())
    rec: Dict[str, Any] = {
        "activity_id": activity_id,
        "user_id": user_id,
        "type": activity_type,
        "status": "PENDING",
        "cost_usd": cost_usd,
        "start_time": _now_iso(),
        "end_time": None,
        "payload": scrub_payload(payload) if payload is not None else None,
        "response_preview": "",
        "metadata": metadata or {}
    }
    _atomic_write(_activity_path(user_id, activity_id), rec)
    return activity_id

def update_activity_status(
    *,
    user_id: str,
    activity_id: str,
    status: ActivityStatus,
    set_cost: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    end: bool = False,
) -> None:
    rec = _load(user_id, activity_id)
    rec["status"] = status
    if set_cost is not None:
        rec["cost_usd"] = set_cost
    if end or status in ("COMPLETED", "ERROR"):
        rec["end_time"] = _now_iso()
    if extra:
        rec.setdefault("metadata", {}).update(scrub_payload(extra))
    _atomic_write(_activity_path(user_id, activity_id), rec)

def append_response_chunk(
    *,
    user_id: str,
    activity_id: str,
    chunk: bytes | str
) -> None:
    rec = _load(user_id, activity_id)
    prev = rec.get("response_preview") or ""
    append_text = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="ignore")
    # limita dimensione (mantiene l'inizio della preview)
    new_prev = (prev + append_text)[:MAX_PREVIEW_BYTES]
    rec["response_preview"] = new_prev
    _atomic_write(_activity_path(user_id, activity_id), rec)

def finalize_activity(
    *,
    user_id: str,
    activity_id: str,
    status: ActivityStatus = "COMPLETED",
    extra: Optional[Dict[str, Any]] = None,
    set_cost: Optional[float] = None
) -> None:
    update_activity_status(
        user_id=user_id,
        activity_id=activity_id,
        status=status,
        set_cost=set_cost,
        extra=extra,
        end=True,
    )

def list_activities(
    *,
    user_id: str,
    filters: Optional[Dict[str, Any]] = None,
    skip: int = 0,
    limit: int = 10
) -> Dict[str, Any]:
    """Restituisce items + total con filtri:
       - date: start_date <= start_time <= end_date   (ISO 8601, UTC)
       - type, status
       - chain_id, upload_task_id, context, file_id, filename, operation (alias di type)
       - q: testo da cercare in payload/response_preview/metadata
    """
    filters = filters or {}
    dirp = _user_dir(user_id)
    items: List[Dict[str, Any]] = []

    # ordina per mtime decrescente per avere le attività più recenti in alto
    for p in sorted(dirp.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            rec = _safe_load_json(p)
        except Exception:
            # se il file è corrotto/illeggibile lo saltiamo (robustezza)
            continue

        # filtri
        def _ok() -> bool:
            sd = filters.get("start_date")
            ed = filters.get("end_date")
            typ = (filters.get("type") or filters.get("operation"))
            st = filters.get("status")
            q = filters.get("q")
            # match semplici
            if typ and rec.get("type") != typ:
                return False
            if st and rec.get("status") != st:
                return False
            # data range (confronto stringhe ISO 8601 UTC coerente)
            st_time = rec.get("start_time")
            if sd and st_time and st_time < sd:
                return False
            if ed and st_time and st_time > ed:
                return False
            # metadati specifici
            md = rec.get("metadata") or {}
            for k in ("chain_id", "upload_task_id", "context", "file_id", "filename"):
                v = filters.get(k)
                if v and md.get(k) != v:
                    return False
            # full-text semplice
            if q:
                blob = json.dumps(
                    {"payload": rec.get("payload"),
                     "response_preview": rec.get("response_preview"),
                     "metadata": md},
                    ensure_ascii=False
                )
                if q.lower() not in blob.lower():
                    return False
            return True

        if _ok():
            items.append(rec)

    total = len(items)
    return {
        "total": total,
        "items": items[skip: skip + limit],
        "skip": skip,
        "limit": limit,
        "total_cost_usd": round(sum((it.get("cost_usd") or 0.0) for it in items), 4)
    }

def update_upload_activity_from_task_status(
    *,
    user_id: str,
    activity_id: str,
    task_status: str
) -> None:
    """Mappa lo stato aggregato dell'upload_task -> stato attività.
       DONE/COMPLETED → COMPLETED
       ERROR          → ERROR
       RUNNING/PENDING→ RUNNING
       altro          → RUNNING (prudente)
    """
    task_status = (task_status or "").upper()
    if task_status in ("DONE", "COMPLETED"):
        finalize_activity(user_id=user_id, activity_id=activity_id, status="COMPLETED")
    elif task_status == "ERROR":
        finalize_activity(user_id=user_id, activity_id=activity_id, status="ERROR")
    elif task_status in ("RUNNING", "PENDING"):
        update_activity_status(user_id=user_id, activity_id=activity_id, status="RUNNING")
    else:
        update_activity_status(user_id=user_id, activity_id=activity_id, status="RUNNING")
