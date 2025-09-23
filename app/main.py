import asyncio
import base64
import copy
from collections.abc import Mapping
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Body, BackgroundTasks, Depends, Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import random
from datetime import datetime
from starlette.responses import StreamingResponse
from io import BytesIO
from pathlib import Path
from PyPDF2 import PdfReader
from pydantic import BaseModel
from pptx import Presentation
import docx
from PIL import Image, TiffImagePlugin
import tempfile
import os, math, json
import httpx
from typing import Literal, Optional, Dict, Any, Union, Annotated
from pydantic import BaseModel, Field
from app.auth_sdk.sdk import CognitoSDK, AccessTokenRequest
from app.payments_sdk.sdk import (
    ApiError,
    MePlansClient,
    DynamicCheckoutRequest,
    PortalSessionRequest,          # <-- nuovo
    PortalConfigSelector,
    PortalUpdateDeepLinkRequest,
    PortalCancelDeepLinkRequest,
    PortalUpgradeDeepLinkRequest,  # <-- nuovo
    RawDiscountSpec,               # <-- nuovo
    ResourcesState,
    DynamicResource,
)
from pydantic import BaseModel, Field, model_validator
from app.system_messages.client_tools_utilities import ToolSpec
from app.system_messages.system_message_1 import get_system_message
from app.utils.payments_utils import (_mk_plans_client, _find_current_subscription_id, _sdk, PLANS_SUCCESS_URL_DEF, \
                                      PLANS_CANCEL_URL_DEF, _variant_to_portal_preset, VARIANTS_BUCKET, RETURN_URL,
                                      _dataclass_to_dict,
                                      build_variants_for_intent, ChangeIntent, features_for_update, _sorted_variants,
                                      _variant_value, _cfg_key, _config_cache_get, _config_cache_put, _price_cache_put,
                                      _price_cache_get, _consume_credits_or_402, PLANS_DEFAULT_PLAN_TYPE)
app = FastAPI(
    root_path="/llm-rag"
)

REQUIRED_AUTH = False

# Crea un'istanza dell'SDK (configura l'URL base secondo le tue necessità)
cognito_sdk = CognitoSDK(base_url="https://teatek-llm.theia-innovation.com/auth")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permetti tutte le origini
    allow_credentials=True,
    allow_methods=["*"],  # Permetti tutti i metodi (GET, POST, OPTIONS, ecc.)
    allow_headers=["*"],  # Permetti tutti gli headers
)

# Carica la configurazione dal file config.json
with open("config.json") as config_file:
    config = json.load(config_file)

NLP_CORE_SERVICE = config["nlp_core_service"]
openai_api_keys = config["openai_api_keys"]


# --- HELPER: filename -> versioni normalizzate -------------------------------
def _safe_filename(name: str) -> str:
    # per mappe loader_*: gli spazi diventano underscore (come già facevi)
    return name.replace(" ", "_")

def _nospace_filename(name: str) -> str:
    # per la collection legacy: gli spazi vengono rimossi
    return name.replace(" ", "_")

# --- HELPER: loader_id deterministico (15 char) ------------------------------
def make_loader_id_from_kwargs(ctx: str, filename: str, effective_kwargs: Mapping) -> tuple[str, str]:
    """
    Ritorna (id_core, loader_id) dove:
      id_core = short_hash({ctx, filename_safe, loader_kwargs}, length=15)
      loader_id = f"{id_core}_loader"
    Include *tutti* i campi dei kwargs (openai_api_key compresa).
    """
    payload = {
        "ctx": ctx,
        "filename": _safe_filename(filename),
        "loader_kwargs": effective_kwargs,  # post-merge; contiene tutti i campi
    }
    id_core = short_hash(payload, length=15)
    loader_id = f"{id_core}_loader"
    return id_core, loader_id

# --- HELPER: collection name legacy -----------------------------------------
def make_legacy_collection_name(ctx: str, filename: str) -> str:
    # esattamente come prima: ctx + filename con spazi rimossi + "_collection"
    return f"{ctx}{_nospace_filename(filename)}_collection"


# ------------------------------------------------------------------
# Vector‑store IDs  ← hash(JSON(cfg))
# ------------------------------------------------------------------
def make_vector_store_ids(cfg: Mapping) -> tuple[str, str]:
    """
    Ritorna (config_id, store_id) deterministici partendo dal JSON
    della configurazione (ordinato).
    """
    cfg_json = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    h = short_hash(cfg_json)                   # 9 caratteri
    return (f"{h}_vector_store_config", h)

def deep_merge(base: Mapping, override: Mapping) -> dict:
    """
    Ritorna un nuovo dict con merge ricorsivo:
      • se la chiave esiste in entrambi e i valori sono dict → merge profondo
      • altrimenti il valore in override ha la precedenza
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(val, Mapping)
        ):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return dict(result)


def short_hash(obj: dict | str, length: int = 9) -> str:
    """
    Restituisce i primi length caratteri dell'hash SHA‑256
    calcolato sull'oggetto (dict o stringa) passato.
    """
    if not isinstance(obj, str):
        obj = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(obj.encode("utf-8")).hexdigest()[:length]

# Models for handling requests and responses
class ContextMetadata(BaseModel):
    path: str
    custom_metadata: Optional[Dict[str, Any]] = None


class FileUploadResponse(BaseModel):
    file_id: str
    contexts: List[str]
    tasks: Optional[Any]=None

# --- aggiungi vicino agli altri BaseModel ---------------------------
class GetChainConfigurationRequest(BaseModel):
    chain_id: str | None = Field(
        default=None,
        description="ID della chain (senza _config). Se presente ha precedenza su chain_config_id."
    )
    chain_config_id: str | None = Field(
        default=None,
        description="ID della configurazione della chain (termina con _config)."
    )
    token: str | None = Field(
        default=None,
        description="Access token (richiesto solo se REQUIRED_AUTH=True)."
    )


# ------------------------------------------------------------------
# NUOVO: descrive la configurazione di un LLM ChatOpenAI
# ------------------------------------------------------------------
class LLMConfig(BaseModel):
    model_name: str = "gpt-4o"          # default
    temperature: float = 0.25
    max_tokens: int = 16000
    max_retries: int = 2
    api_key: Optional[str] = None       # se None → lo riempiamo runtime


def make_llm_ids(cfg: LLMConfig) -> tuple[str, str]:
    """
    Ritorna (config_id, model_id) deterministici
    partendo dal JSON della configurazione.
    """
    cfg_json = json.dumps(cfg.model_dump(), sort_keys=True)
    h = short_hash(cfg_json)                       # 9 caratteri
    return f"{h}_config", h


async def _post_or_400(client, url: str, **kw):
    """POST helper con gestione errore standard."""
    r = await client.post(url, **kw)
    if r.status_code not in (200, 400):
        raise HTTPException(r.status_code, detail=r.text)
    return r


async def _wait_task_done_(client, status_url: str, *, poll_secs: float = 2.0):
    """Attende che lo stato del task diventi DONE o ERROR."""
    while True:
        st = (await client.get(status_url)).json()
        print(st)
        try:
            if st["status"] in ("DONE", "ERROR"):
                if st["status"] == "ERROR":
                    raise HTTPException(500, f"Background task failed: {st['error']}")
                return st
            await asyncio.sleep(poll_secs)
        except Exception as e:
            print(f"[ERROR]: {e}")

import time

async def _wait_task_done(
    client,
    status_url: str,
    *,
    poll_secs: float = 5.0,
    max_wait: float = 1800.0,          # 30 min → scegli tu
):


    start = time.monotonic()
    while True:
        st = (await client.get(status_url)).json()

        if st["status"] in ("DONE", "ERROR"):
            if st["status"] == "ERROR":
                raise HTTPException(500, f"Background task failed: {st['error']}")
            return st

        if time.monotonic() - start > max_wait:
            raise HTTPException(
                504,
                f"Timeout dopo {max_wait}s in attesa che {status_url} uscisse da PENDING"
            )

        await asyncio.sleep(poll_secs)


def _build_loader_config_payload(
    context: str,
    file: UploadFile,
    collection_name: str | None,      # ignorato: usiamo naming legacy
    loader_config_id: str | None,     # ignorato: calcoliamo noi da kwargs
    custom_loaders: Optional[Dict[str, str]] = None,
    custom_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> tuple[str, str, dict]:
    """
    Crea il payload JSON per /document_loaders/configure_loader con:
      • loader_id deterministico (hash 15) da {ctx, filename_norm, chosen_kwargs}
      • coll_name legacy = f"{ctx}{filename_senza_spazi}_collection"
    Ritorna: (loader_id, coll_name, payload)
    """
    file_type = file.filename.split(".")[-1].lower()
    safe_name = _safe_filename(file.filename)

    # mappa estensione → loader (base)
    base_loaders = {
        "png": "ImageDescriptionLoader",
        "jpg": "ImageDescriptionLoader",
        "jpeg": "ImageDescriptionLoader",
        "avi": "VideoDescriptionLoader",
        "mp4": "VideoDescriptionLoader",
        "mov": "VideoDescriptionLoader",
        "mkv": "VideoDescriptionLoader",
        "default": "UnstructuredLoader",
    }

    # kwargs specifici per ciascun loader (base)
    base_kwargs = {
        "png":  {"openai_api_key": get_random_openai_api_key(), "resize_to": (256, 256)},
        "jpg":  {"openai_api_key": get_random_openai_api_key(), "resize_to": (256, 256)},
        "jpeg": {"openai_api_key": get_random_openai_api_key(), "resize_to": (256, 256)},
        "avi":  {"resize_to": [256, 256], "num_frames": 10, "openai_api_key": get_random_openai_api_key()},
        "mp4":  {"resize_to": [256, 256], "num_frames": 10, "openai_api_key": get_random_openai_api_key()},
        "mov":  {"resize_to": [256, 256], "num_frames": 10, "openai_api_key": get_random_openai_api_key()},
        "mkv":  {"resize_to": [256, 256], "num_frames": 10, "openai_api_key": get_random_openai_api_key()},
        "default": {"strategy": "hi_res", "partition_via_api": False},
    }

    # merge profondo con eventuali override
    eff_loaders = deep_merge(base_loaders, custom_loaders or {})
    eff_kwargs  = deep_merge(base_kwargs,  custom_kwargs  or {})

    # selezione finale
    chosen_loader = eff_loaders.get(file_type, eff_loaders["default"])
    chosen_kwargs = eff_kwargs.get(file_type, eff_kwargs["default"])

    # === ID loader deterministico (15 char) ===
    _, computed_loader_id = make_loader_id_from_kwargs(context, file.filename, chosen_kwargs)

    # === Collection name LEGACY (come prima) ===
    computed_coll_name = make_legacy_collection_name(context, file.filename)

    payload = {
        "config_id": computed_loader_id,
        "path": f"data_stores/data/{context}",
        "loader_map": {safe_name: chosen_loader},
        "loader_kwargs_map": {safe_name: chosen_kwargs},
        "metadata_map": {safe_name: {"source_context": context}},
        "default_metadata": {"source_context": context},
        "recursive": True,
        "max_depth": 5,
        "silent_errors": True,
        "load_hidden": True,
        "show_progress": True,
        "use_multithreading": True,
        "max_concurrency": 8,
        "exclude": ["*.tmp", "*.log"],
        "sample_size": 10,
        "randomize_sample": True,
        "sample_seed": 42,
        "output_store_map": {safe_name: {"collection_name": computed_coll_name}},
        "default_output_store": {"collection_name": computed_coll_name},
    }

    return computed_loader_id, computed_coll_name, payload



async def _ensure_vector_store(
    context: str,
    client: httpx.AsyncClient,
) -> tuple[str, str]:
    """
    - genera/configura il vector-store se non esiste
    - lo carica in memoria (endpoint /load)
    - restituisce (vector_store_config_id, vector_store_id)
    """
    '''vector_store_config_id = f"{context}_vector_store_config"
    vector_store_id        = f"{context}_vector_store"

    vector_store_config = {
        "config_id": vector_store_config_id,
        "store_id": vector_store_id,
        "vector_store_class": "Chroma",
        "params": {"persist_directory": f"vector_stores/{context}"},
        "embeddings_model_class": "OpenAIEmbeddings",
        "embeddings_params": {"api_key": get_random_openai_api_key()},
        "description": f"Vector store for context {context}",
        "custom_metadata": {"source_context": context},
    }'''
    base_cfg = {  # ← solo la “sostanza”
                "vector_store_class": "Chroma",
                "params": {"persist_directory": f"vector_stores/{context}"},
                "embeddings_model_class": "OpenAIEmbeddings",
                "embeddings_params": {"api_key": get_random_openai_api_key()},
                "description": f"Vector store for context {context}",
                "custom_metadata": {"source_context": context},
        }

    vector_store_config_id, vector_store_id = make_vector_store_ids(base_cfg)

    vector_store_config = {  # ← ora aggiungiamo gli ID
                "config_id": vector_store_config_id,
                "store_id": vector_store_id,
        **base_cfg
    }

    # --- 1. configure (idempotente) -----------------------------------------
    cfg_resp = await client.post(
        f"{NLP_CORE_SERVICE}/vector_stores/vector_store/configure",
        json=vector_store_config,
    )
    if cfg_resp.status_code not in (200, 400):          # 400 = già esiste
        raise HTTPException(cfg_resp.status_code, cfg_resp.text)

    # --- 2. load in RAM ------------------------------------------------------
    load_resp = await client.post(
        f"{NLP_CORE_SERVICE}/vector_stores/vector_store/load/{vector_store_config_id}"
    )
    if load_resp.status_code not in (200, 400):         # 400 = già caricato
        raise HTTPException(load_resp.status_code, load_resp.text)

    return vector_store_config_id, vector_store_id


'''async def _process_context_pipeline(
    ctx: str,
    file: UploadFile,
    file_content: bytes,
    file_uuid: str,
    file_metadata: dict,
    loader_task_id: str,
    vector_task_id: str,
    client: httpx.AsyncClient,
    loaders: Optional[Dict[str, str]] = None,
    loader_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """
    Esegue TUTTE le operazioni “pesanti” per un singolo context:
        0. upload raw file
        1. config loader  + load_documents_async (attesa DONE)
        2. config vector store (idempotente)      + add_docs_async
    Tutte le chiamate sono già non-bloccanti vs l'utente; qui le orchestriamo.
    """
    # ---------- 0. upload file --------------------------------------------------
    data = {
        "subdir": ctx,
        "extra_metadata": json.dumps({"file_uuid": file_uuid, **file_metadata})
    }
    files = {"file": (file.filename.replace(" ", "_"), file_content, file.content_type)}
    await _post_or_400(client, f"{NLP_CORE_SERVICE}/data_stores/upload", data=data, files=files)

    # ---------- 1. prepare loader ----------------------------------------------
    # Calcolo deterministico degli ID (15 char) a partire da ctx, filename e *chosen* loader_kwargs.
    # NB: la funzione aggiornata restituisce: (loader_id, coll_name, payload)
    loader_id, coll_name, loader_payload = _build_loader_config_payload(
        ctx,
        file,
        collection_name=None,           # calcolata internamente
        loader_config_id=None,          # calcolato internamente
        custom_loaders=loaders,
        custom_kwargs=loader_kwargs,
    )

    print("#" * 120)
    print(loaders)
    print(loader_kwargs)
    print(json.dumps(loader_payload, indent=4))
    print("#" * 120)

    await _post_or_400(
        client,
        f"{NLP_CORE_SERVICE}/document_loaders/configure_loader",
        json=loader_payload
    )

    # lancia il loader in async e aspetta che finisca
    STEP_1 = await _post_or_400(
        client,
        f"{NLP_CORE_SERVICE}/document_loaders/load_documents_async/{loader_id}",
        data={"task_id": loader_task_id},
    )
    print("#"*120)
    print(STEP_1)
    print("#" * 120)

    WAIT_1 = await _wait_task_done(client, f"{NLP_CORE_SERVICE}/document_loaders/task_status/{loader_task_id}")

    print("#"*120)
    print(WAIT_1)
    print("#" * 120)

    # ---------- 2. config / load vector store ----------------------------------
    _, vect_id = await _ensure_vector_store(ctx, client)  # idempotente

    STEP_2 = await _post_or_400(
        client,
        f"{NLP_CORE_SERVICE}/vector_stores/vector_store/add_documents_from_store_async/{vect_id}",
        params={"document_collection": coll_name, "task_id": vector_task_id},
    )
    print("#"*120)
    print(STEP_2)
    print("#" * 120)
    WAIT_2 = await _wait_task_done(
        client,
        f"{NLP_CORE_SERVICE}/vector_stores/vector_store/task_status/{vector_task_id}"
    )
    print("#"*120)
    print(WAIT_2)
    print("#" * 120)'''

async def _process_context_pipeline(
    ctx: str,
    file: UploadFile,
    file_content: bytes,
    file_uuid: str,
    file_metadata: dict,
    loader_task_id: str,
    vector_task_id: str,
    client: httpx.AsyncClient,
    loaders: Optional[Dict[str, str]] = None,
    loader_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """
    Esegue TUTTE le operazioni “pesanti” per un singolo context:
        0. upload raw file
        1. config loader  + load_documents_async (attesa DONE)
        2. config vector store (idempotente)      + add_docs_async
    Tutte le chiamate sono già non-bloccanti vs l'utente; qui le orchestriamo.
    """
    # ---------- 0. upload file --------------------------------------------------
    data  = {
        "subdir": ctx,
        "extra_metadata": json.dumps({"file_uuid": file_uuid, **file_metadata})
    }
    files = {"file": (_safe_filename(file.filename), file_content, file.content_type)}
    await _post_or_400(client, f"{NLP_CORE_SERVICE}/data_stores/upload", data=data, files=files)

    # ---------- 1. prepare loader ----------------------------------------------
    # Calcolo deterministico del loader_id; collection name LEGACY (come prima)
    loader_id, coll_name, loader_payload = _build_loader_config_payload(
        ctx,
        file,
        collection_name=None,           # ignorato (legacy inside)
        loader_config_id=None,          # ignorato (hash inside)
        custom_loaders=loaders,
        custom_kwargs=loader_kwargs,
    )

    print("#"*120)
    print(loaders)
    print(loader_kwargs)
    print(json.dumps(loader_payload, indent=4))
    print("#"*120)

    await _post_or_400(
        client,
        f"{NLP_CORE_SERVICE}/document_loaders/configure_loader",
        json=loader_payload
    )

    # lancia il loader in async e aspetta che finisca
    STEP_1 = await _post_or_400(
        client,
        f"{NLP_CORE_SERVICE}/document_loaders/load_documents_async/{loader_id}",
        data={"task_id": loader_task_id},
    )
    print("#"*120)
    print(STEP_1)
    print("#" * 120)

    WAIT_1 = await _wait_task_done(client, f"{NLP_CORE_SERVICE}/document_loaders/task_status/{loader_task_id}")

    print("#"*120)
    print(WAIT_1)
    print("#" * 120)

    # ---------- 2. config / load vector store ----------------------------------
    _, vect_id = await _ensure_vector_store(ctx, client)  # idempotente

    STEP_2 = await _post_or_400(
        client,
        f"{NLP_CORE_SERVICE}/vector_stores/vector_store/add_documents_from_store_async/{vect_id}",
        params={"document_collection": coll_name, "task_id": vector_task_id},
    )
    print("#"*120)
    print(STEP_2)
    print("#" * 120)
    WAIT_2 = await _wait_task_done(
        client,
        f"{NLP_CORE_SERVICE}/vector_stores/vector_store/task_status/{vector_task_id}"
    )
    print("#"*120)
    print(WAIT_2)
    print("#" * 120)


# --------------------------- REWRITE *ENTIRE* helper ---------------------------
async def upload_file_to_contexts_async(
    file: UploadFile,
    contexts: List[str],
    file_metadata: Optional[Dict[str, Any]] = None,
    *,                       # ← chiamato dall’endpoint /upload
    background_tasks: BackgroundTasks,
    loaders: Optional[Dict[str, str]] = None,
    loader_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
):


    """
    Orchestratore: prepara i task-id, accoda le pipeline in BackgroundTasks
    e restituisce subito la mappa <context → {loader_task_id, vector_task_id}>.
    """
    timeout       = httpx.Timeout(600.0, connect=600.0, read=600.0, write=600.0)
    file_uuid     = str(uuid.uuid4())                      # docs.python.org :contentReference[oaicite:3]{index=3}
    file_content  = await file.read()
    contexts      = contexts[0].split(",")
    task_map: Dict[str, Dict[str, str]] = {}

    # Creiamo UN client condiviso per tutti i task (performance)
    client = httpx.AsyncClient(timeout=timeout)            # httpx docs :contentReference[oaicite:4]{index=4}

    for ctx in contexts:
        loader_task_id = str(uuid.uuid4())
        vector_task_id = str(uuid.uuid4())
        task_map[ctx]  = {
            "loader_task_id": loader_task_id,
            "vector_task_id": vector_task_id,
        }

        # Accoda il worker in background, *niente tracking interno aggiuntivo*.
        background_tasks.add_task(
            _process_context_pipeline,
            ctx,
            file,
            file_content,
            file_uuid,
            file_metadata or {},
            loader_task_id,
            vector_task_id,
            client,
            loaders,
            loader_kwargs# passiamo il client già creato
        )

    # la response torna SUBITO
    return {"file_id": file_uuid, "contexts": contexts, "tasks": task_map}




# Funzione per selezionare una chiave API casuale
def get_random_openai_api_key():
    return random.choice(openai_api_keys)


# Definisci la funzione di verifica in un punto centrale del tuo script
def verify_access_token(token: str | None, sdk_instance) -> None:
    """
    Verifica l'access token usando l'SDK.

    Args:
        token (str): L'access token da verificare.
        sdk_instance: Un'istanza di CognitoSDK (o altro SDK) con il metodo verify_token.

    Raises:
        HTTPException: Se l'access token non è valido.
    """
    #try:
    # L'SDK attende un dizionario con la chiave "access_token"
    print(token)
    _ = sdk_instance.verify_token(AccessTokenRequest(access_token=token))
    #except Exception as e:
    #    raise HTTPException(status_code=401, detail="Access token non valido")

# Helper function to communicate with the existing API
async def create_context_on_server(context_path: str, metadata: Optional[Dict[str, Any]] = None):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{NLP_CORE_SERVICE}/data_stores/create_directory",
            data={
                "directory": context_path,
                "extra_metadata": metadata and json.dumps(metadata)
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.json())
        return response.json()


async def delete_context_on_server(context_path: str):
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{NLP_CORE_SERVICE}/data_stores/delete_directory/{context_path}")
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.json())
        return response.json()




async def list_contexts_from_server(prefix: Optional[str] = None):
    """Chiama /data_stores/directories con eventuale filtro di prefisso."""
    params = {"prefix": prefix} if prefix else None

    timeout = httpx.Timeout(
        connect=30.0,  # Tempo massimo per stabilire una connessione
        read=30.0,  # Tempo massimo per ricevere una risposta
        write=10.0,  # Tempo massimo per inviare dati
        pool=10.0  # Tempo massimo per acquisire una connessione dal pool
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(
            f"{NLP_CORE_SERVICE}/data_stores/directories",
            params=params,
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.json())
        return response.json()


async def upload_file_to_contexts_(file: UploadFile, contexts: List[str],
                                   file_metadata: Optional[Dict[str, Any]] = None):
    file_uuid = str(uuid.uuid4())  # Generate a UUID for the file
    file_content = await file.read()  # Read the file content once and reuse it

    contexts = contexts[0].split(',')

    async with httpx.AsyncClient() as client:
        responses = []
        # Sequentially upload the file to each context
        for context in contexts:
            # Ensure that each context is handled separately and not concatenated
            data = {
                "subdir": context,  # Here, context is passed as a single string, not a concatenation
                "extra_metadata": json.dumps({"file_uuid": file_uuid, **(file_metadata or {})}),
            }
            files = {"file": (file.filename.replace(" ", "_"), file_content, file.content_type)}

            # Make the POST request to upload the file to the current context
            response = await client.post(f"{NLP_CORE_SERVICE}/data_stores/upload", data=data, files=files)

            # Log and handle errors
            if response.status_code != 200:
                print(
                    f"Error uploading to {context}. Status Code: {response.status_code}. Response content: {response.content}")

                try:
                    error_detail = response.json()
                except ValueError:
                    raise HTTPException(status_code=response.status_code, detail=f"Error: {response.text}")

                raise HTTPException(status_code=response.status_code, detail=error_detail)

            # Collect response data for successful uploads
            try:
                responses.append(response.json())
            except ValueError:
                raise HTTPException(status_code=500, detail="Received invalid JSON response from the server")

        # Return the collected responses with file UUID and associated contexts
        return {"file_id": file_uuid, "contexts": contexts}


async def upload_file_to_contexts(file: UploadFile,
                                  contexts: List[str],
                                  file_metadata: Optional[Dict[str, Any]] = None,
                                  loaders: Optional[Dict[str, str]] = None,
                                  loader_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
                                  ):
    file_uuid = str(uuid.uuid4())  # Generate a UUID for the file
    file_content = await file.read()  # Read the file content once and reuse it

    contexts = contexts[0].split(',')
    timeout_settings = httpx.Timeout(600.0, connect=600.0, read=600.0, write=600.0)
    async with httpx.AsyncClient() as client:
        responses = []
        # Sequentially upload the file to each context
        for context in contexts:
            # Upload the file
            data = {
                "subdir": context,
                "extra_metadata": json.dumps({"file_uuid": file_uuid, **(file_metadata or {})}),
            }
            files = {"file": (file.filename.replace(" ", "_"), file_content, file.content_type)}

            # Make the POST request to upload the file to the current context
            response = await client.post(f"{NLP_CORE_SERVICE}/data_stores/upload", data=data, files=files,
                                         timeout=timeout_settings)

            if response.status_code != 200:
                print(
                    f"Error uploading to {context}. Status Code: {response.status_code}. Response content: {response.content}")
                try:
                    error_detail = response.json()
                except ValueError:
                    raise HTTPException(status_code=response.status_code, detail=f"Error: {response.text}")

                raise HTTPException(status_code=response.status_code, detail=error_detail)

            # Collect response data for successful uploads
            try:
                upload_response = response.json()
                responses.append(upload_response)
            except ValueError:
                raise HTTPException(status_code=500, detail="Received invalid JSON response from the server")

            # Configure the loader for the uploaded file
            loader_config_id = f"{context}{file.filename.replace(' ', '')}_loader"
            doc_store_collection_name = f"{context}{file.filename.replace(' ', '')}_collection"

            file_type = file.filename.split(".")[-1].lower()

            loaders = {
                # "pdf": "PyMuPDFLoader",
                # "txt": "TextLoader",
                "png": "ImageDescriptionLoader",
                "jpg": "ImageDescriptionLoader",
                "jpeg": "ImageDescriptionLoader",
                "avi": "VideoDescriptionLoader",
                "mp4": "VideoDescriptionLoader",
                "mov": "VideoDescriptionLoader",
                "mkv": "VideoDescriptionLoader",
                "default": "UnstructuredLoader"
            }

            kwargs = {
                # "pdf": {
                # "pages": None,
                # "page_chunks": True,
                # "write_images": False,
                # "image_size_limit": 0.025,
                # "embed_images": True,
                # "image_path": "C:\\Users\\Golden Bit\\Desktop\\projects_in_progress\\GoldenProjects\\golden_bit\\repositories\\nlp-core-api\\tmp",
                # },
                "png": {
            "openai_api_key": get_random_openai_api_key(),
            "resize_to": (1024, 1024),            # ↑ higher baseline
            #"description_mode": "extended",     # ← NEW
        },
                "jpg": {
            "openai_api_key": get_random_openai_api_key(),
            "resize_to": (1024, 1024),            # ↑ higher baseline
            #"description_mode": "extended",     # ← NEW
        },
                # "txt": {}
                "avi": {
            "resize_to": [1024, 1024],
            "frame_rate": 0.3,                  # ← sample at 0.3 fps
            #"description_mode": "extended",
            "openai_api_key": get_random_openai_api_key(),
        },
                "mp4": {
            "resize_to": [1024, 1024],
            "frame_rate": 0.3,                  # ← sample at 0.3 fps
            #"description_mode": "extended",
            "openai_api_key": get_random_openai_api_key(),
        },
                "mov": {
            "resize_to": [1024, 1024],
            "frame_rate": 0.3,                  # ← sample at 0.3 fps
            #"description_mode": "extended",
            "openai_api_key": get_random_openai_api_key(),
        },
                "default": {
                    "strategy": "hi_res",
                    "partition_via_api": False
                }
            }

            loader_config_data = {
                "config_id": loader_config_id,
                "path": f"data_stores/data/{context}",
                "loader_map": {
                    f"{file.filename.replace(' ', '_')}": loaders.get(file_type) or loaders["default"]
                },
                "loader_kwargs_map": {
                    f"{file.filename.replace(' ', '_')}": kwargs.get(file_type) or kwargs["default"]
                },
                "metadata_map": {
                    f"{file.filename.replace(' ', '_')}": {
                        "source_context": f"{context}"
                    }
                },
                "default_metadata": {
                    "source_context": f"{context}"
                },
                "recursive": True,
                "max_depth": 5,
                "silent_errors": True,
                "load_hidden": True,
                "show_progress": True,
                "use_multithreading": True,
                "max_concurrency": 8,
                "exclude": [
                    "*.tmp",
                    "*.log"
                ],
                "sample_size": 10,
                "randomize_sample": True,
                "sample_seed": 42,
                "output_store_map": {
                    f"{file.filename.replace(' ', '_')}": {
                        "collection_name": doc_store_collection_name
                    }
                },
                "default_output_store": {
                    "collection_name": doc_store_collection_name
                }
            }

            # Configure the loader on the original API
            loader_response = await client.post(f"{NLP_CORE_SERVICE}/document_loaders/configure_loader",
                                                json=loader_config_data)
            print(loader_response.json())
            if loader_response.status_code != 200 and loader_response.status_code != 400:
                raise HTTPException(status_code=loader_response.status_code, detail=loader_response.json())

            # Apply the loader to process the document
            load_response = await client.post(f"{NLP_CORE_SERVICE}/document_loaders/load_documents/{loader_config_id}",
                                              timeout=timeout_settings)

            print(load_response)
            if load_response.status_code != 200:
                raise HTTPException(status_code=load_response.status_code, detail=load_response.json())

            # Collect document processing results
            # processed_docs = load_response.json()

            ### Configure the Vector Store ###

            vector_store_config_id = f"{context}_vector_store_config"
            vector_store_id = f"{context}_vector_store"

            vector_store_config = {
                "config_id": vector_store_config_id,
                "store_id": vector_store_id,
                "vector_store_class": "Chroma",  # Example: using Chroma vector store, modify as necessary
                "params": {
                    "persist_directory": f"vector_stores/{context}"
                },
                "embeddings_model_class": "OpenAIEmbeddings",
                "embeddings_params": {
                    "api_key": get_random_openai_api_key()  # Seleziona una chiave API casuale
                },
                "description": f"Vector store for context {context}",
                "custom_metadata": {
                    "source_context": context
                }
            }

            # Configure the vector store
            vector_store_response = await client.post(
                f"{NLP_CORE_SERVICE}/vector_stores/vector_store/configure", json=vector_store_config,
                timeout=timeout_settings)
            if vector_store_response.status_code != 200 and vector_store_response.status_code != 400:
                raise HTTPException(status_code=vector_store_response.status_code, detail=vector_store_response.json())

            # vector_store_config_id = vector_store_response.json()["config_id"]

            cnt = 0
            while True:
                cnt += 1
                ### Load the Vector Store ###
                load_vector_response = await client.post(
                    f"{NLP_CORE_SERVICE}/vector_stores/vector_store/load/{vector_store_config_id}",
                    timeout=timeout_settings)
                if load_vector_response.status_code != 200 and load_vector_response.status_code != 400:
                    raise HTTPException(status_code=load_vector_response.status_code,
                                        detail=load_vector_response.json())

                ### Add Documents from the Document Store to the Vector Store ###
                # Use the document collection name associated with the context
                add_docs_response = await client.post(
                    f"{NLP_CORE_SERVICE}/vector_stores/vector_store/add_documents_from_store/{vector_store_id}",
                    params={"document_collection": doc_store_collection_name}, timeout=timeout_settings)
                if add_docs_response.status_code != 200:
                    print(add_docs_response.content)
                    if cnt > 5:
                        raise HTTPException(status_code=add_docs_response.status_code, detail=add_docs_response.json())
                else:
                    break

        # Return the collected responses with file UUID and associated contexts
        return {"file_id": file_uuid, "contexts": contexts}


########################################################################################################################
########################################################################################################################

# ----------------------------------------------------------------------
# UPDATE helpers
# ----------------------------------------------------------------------

async def update_context_metadata_on_server(
    context_path: str,
    description: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
):
    """Chiama PUT /data_stores/directory/metadata/{context_path}"""
    payload = {}
    if description is not None:
        payload["description"] = description
    if extra_metadata is not None:
        payload["extra_metadata"] = json.dumps(extra_metadata)

    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{NLP_CORE_SERVICE}/data_stores/directory/metadata/{context_path}",
            data=payload,
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, detail=response.json()
            )
        return response.json()


async def update_file_metadata_on_server(
    file_path: str,
    description: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
):
    """Chiama PUT /data_stores/file/metadata/{file_path}"""
    payload = {}
    if description is not None:
        payload["file_description"] = description
    if extra_metadata is not None:
        payload["extra_metadata"] = json.dumps(extra_metadata)

    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{NLP_CORE_SERVICE}/data_stores/file/metadata/{file_path}",
            data=payload,
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, detail=response.json()
            )
        return response.json()


# Create a new context (directory)
class CreateContextRequest(BaseModel):
    username: str
    token: str
    context_name: str          # UUID
    display_name: str          # ⬅️ nuovo
    description: Optional[str] = None
    extra_metadata: Optional[Any] = None

class UpdateContextMetadataRequest(BaseModel):
    username: str
    token: Optional[str] = None
    context_name: str
    description: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None


class UpdateFileMetadataRequest(BaseModel):
    token: Optional[str] = None
    # uno dei due:
    file_path: Optional[str] = None   # path completo "username-context/filename"
    file_id: Optional[str] = None     # UUID da custom_metadata.file_uuid
    description: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None


@app.post("/contexts", response_model=ContextMetadata)
async def create_context(request: CreateContextRequest):

    print(request.model_dump_json(indent=2))

    if REQUIRED_AUTH:
        verify_access_token(request.token, cognito_sdk)

    username = request.username

    print(f"Creating context: {request.context_name} for user: {username}")

    # Aggiungi username nei metadati del contesto
    metadata = {
        "display_name": request.display_name,
        "description": request.description,
        # "owner": username  # Memorizziamo l'username dell'utente che ha creato il contesto
    }  # if request.description else {"owner": username}

    if request.extra_metadata:
        metadata.update(request.extra_metadata)

    result = await create_context_on_server(f"{username}-{request.context_name}", metadata)

    return result


# Delete an existing context (directory)
@app.delete("/contexts/{context_name}", response_model=Dict[str, Any])
async def delete_context(context_name: str, token: str=None):

    if REQUIRED_AUTH:
        verify_access_token(token, cognito_sdk)

    result = await delete_context_on_server(context_name)
    # TODO: delete related vector store (and all related collection in document store)
    return result


# List all available contexts
class ListContextsRequest(BaseModel):
    username: str
    token: str = None

@app.post("/list_contexts", response_model=List[ContextMetadata])
async def list_contexts(request: ListContextsRequest):
    username = request.username

    # Recupera SOLO i contesti che iniziano con "<username>-"
    all_contexts = await list_contexts_from_server(prefix=f"{username}-")

    # Rimuovi il prefisso prima di restituire i path all’utente
    user_contexts = [
        ContextMetadata(
            path=ctx["path"].removeprefix(f"{username}-"),
            custom_metadata=ctx.get("custom_metadata"),
        )
        for ctx in all_contexts
    ]

    for ctx in all_contexts:

        if "display_name" not in ctx["custom_metadata"]:
            print(ctx)
            print('#'*120)

    if not user_contexts:
        raise HTTPException(status_code=403, detail="Non sei autorizzato a visualizzare questi contesti.")

    return user_contexts


"""
@app.post("/list_contexts", response_model=List[ContextMetadata])
async def list_contexts(request: ListContextsRequest):

    if REQUIRED_AUTH:
        verify_access_token(request.token, cognito_sdk)

    username = request.username

    print(f"Listing contexts for user: {username}")

    # Otteniamo tutti i contesti dal server
    all_contexts = await list_contexts_from_server()

    # Filtriamo i contesti per verificare che l'utente sia il proprietario
    user_contexts = []
    for context in all_contexts:
        print(context)
        path = context.get("path")  # Estrai l'username del proprietario
        print(path)
        if path:
            if path.startswith(f"{username}-"):
                context["path"] = context["path"].removeprefix(f"{username}-")
                user_contexts.append(context)
            else:
                print(f"User {username} is not authorized to access")

    # Se l'utente non ha accesso a nessun contesto, restituisci errore
    if not user_contexts:
        raise HTTPException(status_code=403, detail="Non sei autorizzato a visualizzare questi contesti.")

    return user_contexts
"""

# Upload a file to multiple contexts
@app.post("/upload", response_model=FileUploadResponse)
async def upload_file_to_multiple_contexts(
        file: UploadFile = File(...),
        contexts: List[str] = Form(...),
        description: Optional[str] = Form(None),
        extra_metadata: Optional[Any] = Form(None),
        username: Optional[str] = Form(None),
        loaders: Optional[str] = Form(None),
        loader_kwargs: Optional[str] = Form(None),
        token: Optional[str] = Form(None),
        subscription_id: Optional[str] = Form(None),
):

    if REQUIRED_AUTH:
        verify_access_token(token, cognito_sdk)

    ####################################################################################################################
    # TODO:
    #  - verifica se in path è presente prefisso, altrimenti aggiungi (dovremo richiedere anche username in input)

    if username:
        formatted_contexts = []

        for context in contexts:
            if not context.startswith(f"{username}-"):
                context = f"{username}-{context}"
            formatted_contexts.append(context)

        contexts = formatted_contexts

    ####################################################################################################################

    try:
        loaders_dict: Dict[str, str] | None = json.loads(loaders) if loaders else None
        kwargs_dict: Dict[str, Dict[str, Any]] | None = (
            json.loads(loader_kwargs) if loader_kwargs else None
        )
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"Parametri JSON non validi: {e}")

    file_metadata = {"description": description} if description else None

    if extra_metadata:
        file_metadata.update(extra_metadata)

    # ============================================================
    # 1) STIMA COSTO (riusiamo direttamente l’endpoint-funzione)
    #    NB: estimate_file_processing_cost consuma lo stream -> reset poi
    # ============================================================
    est = await estimate_file_processing_cost(
        files=[file],
        loader_kwargs=(loader_kwargs or None)  # passiamo la stringa originale
    )
    # il model pydantic è già serializzato: estraiamo il totale
    credits_to_consume = est.grand_total

    # Riposiziona lo stream del file, altrimenti l’upload leggerà 0 byte
    try:
        await file.seek(0)
    except Exception:
        if hasattr(file, "file"):
            file.file.seek(0)

    # ============================================================
    # 2) CONSUMO CREDITI
    # ============================================================
    await _consume_credits_or_402(
        token,
        credits_to_consume,
        reason=f"upload+processing file={file.filename} contexts={contexts}",
        subscription_id=subscription_id
    )


    result = await upload_file_to_contexts(
        file,
        contexts,
        file_metadata,
        loaders=loaders_dict,
        loader_kwargs=kwargs_dict
    )

    return result


@app.post("/upload_async", response_model=FileUploadResponse)
async def upload_file_to_multiple_contexts_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    contexts: List[str] = Form(...),
    description: Optional[str] = Form(None),
    username: Optional[str] = Form(None),
    loaders: Optional[str] = Form(None),
    loader_kwargs: Optional[str] = Form(None),
    token: Optional[str] = Form(None),
    subscription_id: Optional[str] = Form(None),
):

    #print("*" * 120)
    #print(token)
    #print("*" * 120)

    if REQUIRED_AUTH:
        verify_access_token(token, cognito_sdk)

    #print("#" * 120)
    #print(json.dumps(loaders, indent=2))
    #print("#"*120)
    #print(json.dumps(loader_kwargs, indent=2))
    #print("#" * 120)

    ####################################################################################################################
    # TODO:
    #  - verifica se in path è presente prefisso, altrimenti aggiungi (dovremo richiedere anche username in input)

    if username:
        formatted_contexts = []

        for context in contexts:
            if not context.startswith(f"{username}-"):
                context = f"{username}-{context}"
            formatted_contexts.append(context)

        contexts = formatted_contexts

    ####################################################################################################################

    try:
        loaders_dict: Dict[str, str] | None = json.loads(loaders) if loaders else None
        kwargs_dict: Dict[str, Dict[str, Any]] | None = (
            json.loads(loader_kwargs) if loader_kwargs else None
        )
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"Parametri JSON non validi: {e}")

    file_meta = {"description": description} if description else None

    # ============================================================
    # === 1) STIMA COSTO (riuso funzione endpoint) ===
    est = await estimate_file_processing_cost(
        files=[file],
        loader_kwargs=(loader_kwargs or None)
    )
    credits_to_consume = est.grand_total

    # reset stream per i task successivi
    try:
        await file.seek(0)
    except Exception:
        if hasattr(file, "file"):
            file.file.seek(0)

    # === 2) CONSUMO CREDITI ===
    await _consume_credits_or_402(
        token,
        credits_to_consume,
        reason=f"upload_async+processing file={file.filename} contexts={contexts}",
        subscription_id=subscription_id
    )

    return await upload_file_to_contexts_async(
        file,
        contexts,
        file_meta,
        background_tasks=background_tasks,
        loaders=loaders_dict,
        loader_kwargs=kwargs_dict,
    )


# Helper function to list files by context
async def list_files_in_context(contexts: Optional[List[str]] = None):
    timeout = httpx.Timeout(
        connect=10.0,  # Tempo massimo per stabilire una connessione
        read=30.0,  # Tempo massimo per ricevere una risposta
        write=10.0,  # Tempo massimo per inviare dati
        pool=10.0  # Tempo massimo per acquisire una connessione dal pool
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        if contexts:
            # If contexts are provided, filter files by those contexts
            files = []
            for context in contexts:
                response = await client.get(f"{NLP_CORE_SERVICE}/data_stores/files", params={"subdir": context})
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=response.json())
                files.extend(response.json())
            return files
        else:
            # No context specified, list all files across all contexts
            response = await client.get(f"{NLP_CORE_SERVICE}/data_stores/files")
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.json())
            return response.json()


# Helper function to delete files by UUID
async def delete_file_by_id(file_id: str):
    async with httpx.AsyncClient() as client:
        # List all contexts to find where the file exists
        response = await client.get(f"{NLP_CORE_SERVICE}/data_stores/files")
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.json())

        # Delete the file from all contexts where the UUID matches
        files = response.json()
        for file in files:
            if file['custom_metadata'].get('file_uuid') == file_id:
                path = file['path']
                delete_response = await client.delete(f"{NLP_CORE_SERVICE}/data_stores/delete/{path}")
                if delete_response.status_code != 200:
                    raise HTTPException(status_code=delete_response.status_code, detail=delete_response.json())
        return {"detail": f"File with ID {file_id} deleted from all contexts"}


# Helper function to delete file by path
async def delete_file_by_path(file_path: str):
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{NLP_CORE_SERVICE}/delete/data_stores/{file_path}")
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.json())
        return {"detail": f"File at path {file_path} deleted successfully"}


# Endpoint to list files by specific context(s)
@app.get("/files", response_model=List[Dict[str, Any]])
async def list_files(
        contexts: Optional[List[str]] = Query(None),
        token: str = Query(None, description="Access token dell'utente"),):
    """
    List files for specific contexts. If no contexts are provided, list all files.
    """

    if REQUIRED_AUTH:
        verify_access_token(token, cognito_sdk)

    result = await list_files_in_context(contexts)
    return result


# Endpoint to delete files by either UUID (deletes from all contexts) or path (deletes from a specific context)
@app.delete("/files")
async def delete_file(
        file_id: Optional[str] = Query(None),
        file_path: Optional[str] = Query(None),
        token: str = Query(None, description="Access token dell'utente")):
    """
    Delete a file by either its UUID (from all contexts) or its path (from a specific context).
    """

    if REQUIRED_AUTH:
        verify_access_token(token, cognito_sdk)

    if file_id:
        # Delete by UUID from all contexts
        result = await delete_file_by_id(file_id)
    elif file_path:
        # Delete by path from a specific context
        result = await delete_file_by_path(file_path)
    else:
        raise HTTPException(status_code=400, detail="Either file_id or file_path must be provided")

    return result

@app.put("/contexts/metadata", response_model=ContextMetadata)
async def update_context_metadata(request: UpdateContextMetadataRequest):
    """
    Aggiorna (merge) i metadati di un *context* (directory).
    """
    if REQUIRED_AUTH:
        verify_access_token(request.token, cognito_sdk)

    # Il percorso reale sul core = "<username>-<context_name>"
    context_path = f"{request.username}-{request.context_name}"
    result = await update_context_metadata_on_server(
        context_path=context_path,
        description=request.description,
        extra_metadata=request.extra_metadata,
    )

    # Nella risposta rimuoviamo il prefisso "<username>-"
    result["path"] = result["path"].removeprefix(f"{request.username}-")
    return result

@app.put("/files/metadata", response_model=Dict[str, Any])
async def update_file_metadata(
    request: UpdateFileMetadataRequest
):
    """
    Aggiorna (merge) i metadati di un file.
    - se file_path è fornito => aggiorna solo quel percorso
    - se file_id (UUID) è fornito => aggiorna tutte le copie di quel file nei vari contesti
    Almeno uno dei due parametri è obbligatorio.
    """
    if REQUIRED_AUTH:
        verify_access_token(request.token, cognito_sdk)

    if not request.file_path and not request.file_id:
        raise HTTPException(
            status_code=400,
            detail="Devi fornire file_path oppure file_id",
        )

    updated_items: List[Dict[str, Any]] = []

    # --- caso 1: percorso esplicito -------------------------------------------------
    if request.file_path:
        meta = await update_file_metadata_on_server(
            file_path=request.file_path,
            description=request.description,
            extra_metadata=request.extra_metadata,
        )
        updated_items.append(meta)

    # --- caso 2: UUID globale -------------------------------------------------------
    elif request.file_id:
        # cerchiamo tutti i file con quel UUID e li aggiorniamo uno ad uno
        files = await list_files_in_context()  # prende tutti i file
        target_files = [
            f for f in files if f["custom_metadata"].get("file_uuid") == request.file_id
        ]
        if not target_files:
            raise HTTPException(status_code=404, detail="File UUID non trovato")

        for f in target_files:
            meta = await update_file_metadata_on_server(
                file_path=f["path"],
                description=request.description,
                extra_metadata=request.extra_metadata,
            )
            updated_items.append(meta)

    return {"updated": updated_items}


# @app.post("/configure_and_load_chain/")
async def configure_and_load_chain_(
        context: str = Query("default", title="Context", description="The context for the chain configuration"),
        model_name: str = Query("gpt-4o-mini", title="Model Name",
                                description="The name of the LLM model to load, default is gpt-4o")
):
    """
    Configura e carica una chain in memoria basata sul contesto dato.
    """

    timeout_settings = httpx.Timeout(600.0, connect=600.0, read=600.0, write=600.0)

    # vector_store_config_id = f"{context}_vector_store_config"
    vector_store_id = f"{context}_vector_store"

    # Impostazione di configurazione per l'LLM basata su model_name (di default "gpt-4o")
    llm_config_id = f"chat-openai_{model_name}_config"
    llm_id = f"chat-openai_{model_name}"

    async with httpx.AsyncClient() as client:
        # 1. Caricamento dell'LLM
        load_llm_url = f"{NLP_CORE_SERVICE}/llms/load_model/{llm_config_id}"
        llm_response = await client.post(load_llm_url, timeout=timeout_settings)

        if llm_response.status_code != 200 and llm_response.status_code != 400:
            raise HTTPException(status_code=llm_response.status_code,
                                detail=f"Errore caricamento LLM: {llm_response.text}")

        llm_load_result = llm_response.json()

    vectorstore_ids = [vector_store_id]

    tools = [{"name": "VectorStoreTools", "kwargs": {"store_id": vectorstore_id}} for vectorstore_id in vectorstore_ids]
    tools.append({"name": "MongoDBTools",
                  "kwargs": {
                      "connection_string": "mongodb://localhost:27017",
                      "default_database": f"{context}_db",
                      "default_collection": "file_descriptions"
                  }})

    # Configurazione della chain
    chain_config = {
        "chain_type": "agent_with_tools",
        "config_id": f"{context}_agent_with_tools_config",
        "chain_id": f"{context}_agent_with_tools",
        "system_message": """

        WRITE HERE SYSTEM MESSAGE...

        """,
        "llm_id": llm_id,  # Usa l'ID del modello LLM configurato
        "tools": tools
    }

    async with httpx.AsyncClient() as client:
        try:
            # 1. Configura la chain
            configure_url = f"{NLP_CORE_SERVICE}/chains/configure_chain/"
            configure_response = await client.post(configure_url, json=chain_config)

            if configure_response.status_code != 200 and configure_response.status_code != 400:
                raise HTTPException(status_code=configure_response.status_code,
                                    detail=f"Errore configurazione: {configure_response.text}")

            configure_result = configure_response.json()

            # 2. Carica la chain
            load_url = f"{NLP_CORE_SERVICE}/chains/load_chain/{chain_config['config_id']}"
            load_response = await client.post(load_url)

            if load_response.status_code != 200 and load_response.status_code != 400:
                raise HTTPException(status_code=load_response.status_code,
                                    detail=f"Errore caricamento: {load_response.text}")

            load_result = load_response.json()

            return {
                "message": "Chain configurata e caricata con successo.",
                "llm_load_result": llm_load_result,
                "config_result": configure_result,
                "load_result": load_result
            }

        except httpx.HTTPStatusError as e:
            print(e)
            raise HTTPException(status_code=e.response.status_code, detail=f"Errore HTTP: {e.response.text}")

        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=f"Errore interno: {str(e)}")


# Modello di input per la configurazione e il caricamento della chain
class ConfigureAndLoadChainInput(BaseModel):
    contexts: List[str] = []  # Lista di contesti (vuota di default)
    llm: Optional[LLMConfig] = None
    model_name: Optional[str] = "gpt-4o"  # Nome del modello, default "gpt-4o-mini"
    system_message: Optional[str] = "You are an helpful assistant."
    system_message_content: Optional[str] = Field(
        default = None,
        description = "Se fornito, verrà aggiunto in coda al sistema di default sotto una sezione 'Additional Instructions'.")
    custom_server_tools: List[Dict[str, Any]] = Field(
        default_factory = list,
        description = 'Lista di tool ({"name":..., "kwargs":{...}}) da aggiungere o con cui sovrascrivere quelli di default.')
    client_tool_specs : List[ToolSpec] = Field(
        default_factory=list,
        description="Elenco facoltativo di tool/widget da documentare"
    )
    token: Optional[str] = None

# -----------------------------------------------------------------------------
# Costruzione “base” e bootstrap di un vector‑store (config + load)
# -----------------------------------------------------------------------------
def _build_vs_base_cfg(ctx: str, api_key: str) -> dict:
    """Restituisce la cfg minima (usata anche per l’hash)."""
    return {
        "vector_store_class": "Chroma",
        "params": {"persist_directory": f"vector_stores/{ctx}"},
        "embeddings_model_class": "OpenAIEmbeddings",
        "embeddings_params": {"api_key": api_key},
        "description": f"Vector store for context {ctx}",
        "custom_metadata": {"source_context": ctx},
    }


async def _ensure_vs_exists(
    ctx: str,
    api_key: str,
    client: httpx.AsyncClient,
) -> str:
    """
    • se il VS (cfg + load) esiste già ⇒ lo carica (idempotente)
    • altrimenti lo configura e poi lo carica
    → ritorna sempre lo **store_id** da usare negli strumenti.
    """

    base_cfg = _build_vs_base_cfg(ctx, api_key)
    cfg_id, store_id = make_vector_store_ids(base_cfg)

    vs_cfg = {"config_id": cfg_id, "store_id": store_id, **base_cfg}

    # 1. configure  (200 = creato, 400 = già presente)
    resp = await client.post(
        f"{NLP_CORE_SERVICE}/vector_stores/vector_store/configure",
        json=vs_cfg,
    )
    if resp.status_code not in (200, 400):

        raise HTTPException(resp.status_code, resp.text)

    # 2. load in RAM  (200 = ok, 400 = già in RAM)
    resp = await client.post(
        f"{NLP_CORE_SERVICE}/vector_stores/vector_store/load/{cfg_id}"
    )

    if resp.status_code not in (200, 400):
        raise HTTPException(resp.status_code, resp.text)

    return store_id

#TODO:
# - assicurarsi che tutti gli oggetti (inclusi vec sotre e llm) siano configurati e caricati usando id derivati da ash della
#   stringa json della configurazione
@app.post("/configure_and_load_chain/")
async def configure_and_load_chain(
        input_data: ConfigureAndLoadChainInput  # Usa il modello come input
):
    """
    Configura e carica una chain in memoria basata sul contesto dato.
    """

    #print(input_data.model_dump_json(indent=2))

    if REQUIRED_AUTH:
        verify_access_token(input_data.token, cognito_sdk)



    client_tool_specs = [t.build_widget_instructions() for t in input_data.client_tool_specs]



    # ------------------------------------------------------------------
    # ● STEP 1 – costruiamo la cfg LLM (default + override da utente)
    # ------------------------------------------------------------------
    model_name = input_data.model_name
    llm_cfg = input_data.llm or LLMConfig()  # default se None
    if model_name: llm_cfg.model_name = model_name

    if llm_cfg.api_key is None:  # riempi API-key on-the-fly
        llm_cfg.api_key = get_random_openai_api_key()

    llm_config_id, llm_id = make_llm_ids(llm_cfg)  # ID deterministici

    # ------------------------------------------------------------------
    # ● STEP 2 – configuriamo il modello (idempotente)
    # ------------------------------------------------------------------
    llm_payload = {
        "config_id": llm_config_id,
        "model_id": llm_id,
        "model_type": "ChatOpenAI",
        "model_kwargs": llm_cfg.model_dump(),
    }

    timeout = httpx.Timeout(600.0, connect=600.0, read=600.0, write=600.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # configure_model (potrebbe già esistere → 400 = OK)
        cfg_resp = await client.post(f"{NLP_CORE_SERVICE}/llms/configure_model/", json=llm_payload)
        if cfg_resp.status_code not in (200, 400):

            raise HTTPException(cfg_resp.status_code, cfg_resp.text)

        # load_model (potrebbe già essere in RAM → 400 = OK)
        llm_load_result = await client.post(f"{NLP_CORE_SERVICE}/llms/load_model/{llm_config_id}")
        if llm_load_result.status_code not in (200, 400):

            raise HTTPException(llm_load_result.status_code, llm_load_result.text)

    llm_load_result = llm_load_result.json()


    # Estrai i valori dal modello
    contexts = input_data.contexts

    '''###########################################
    input_data.system_message = SYSTEM_MESSAGE#
    ###########################################

    system_message = input_data.system_message'''

    system_message = get_system_message(client_tools_instructions=client_tool_specs)

    #id_ = "".join(contexts)

    #id_ = input_data

    # Conversione dell'oggetto in una stringa JSON
    # sort_keys=True garantisce che le chiavi siano ordinate in modo deterministico
    json_str = input_data.model_dump_json()

    # Calcolo dell'hash SHA-256 della stringa JSON
    hash_object = hashlib.sha256(json_str.encode('utf-8'))
    id_ = hash_object.hexdigest()

    #id_ = short_hash(json_str)
    id_ = short_hash(str(uuid.uuid4()))

    timeout_settings = httpx.Timeout(600.0, connect=600.0, read=600.0, write=600.0)

    ####################################################################################################################
    # TODO:
    #  - ottenere i metadati dei contesti
    #  - ottenere lista dei file per i contesti
    # --- Nuovo: recupera metadata e file per ciascun context ---
    contexts_data = []



    for ctx in contexts:
        print(ctx)
        # metadata del context
        ctx_meta = await get_context_info(ctx, input_data.token)
        # file associati al context
        files = await list_files_in_context([ctx])
        contexts_data.append({
                "kbox_id": ctx,
                "metadata": ctx_meta,
                "files": files,
                "vectorstore_id": f"{ctx}_vector_store"
        })

    system_message += f"""
    ## ISTRUZIONI PER L'UTILIZZO DELLE BASI DI CONOSCENZA
    -----------------------------------------------------------------------------------------------------
    - Le knowledge box sono elementi nel quale si caricano file di vario tipo (mostrati dal campo 'files'. Tali files vengono poi processati ed irelativi chunks di testo ottenuti sono poi salvati nel vector store associato e indicato dal campo 'vectorstore_id'.
    - Quando l'utente ti chiede 'cosa conosci' oppure 'di ocsa parla la tua base di conoscenza', allora dovrai effettuare una ricerca nei vectorstore assoiciati alle tue basi di conoscenza usando l apposito strumento (con query esplorative usate per comprenderne il contenuto).
    - I documenti ni vector store possiedono il metadato 'filename', puoi sfruttare tale campo e la conoscenza sui nomi dei file sorgenti al fine di effettuare ricerche con filtro nel vectorstore, ad esmepio per cercare semanticamente solo tr ai docs associati ad uno specifico file sorgente.
    - In generle ogni qual volta che interagisci con l'utente, valuta se si rende utile effettuare ricerche nel vector store, inoltre se necessario effettua più tentativi di ricerca, aggiustando la query o i filtri se necessario.    
    -----------------------------------------------------------------------------------------------------
    LE TUE KNOWLEDGE BOXES:
    {json.dumps(contexts_data, indent=4).replace('{', '{{').replace('}', '}}')}
    -----------------------------------------------------------------------------------------------------
    """

    # ── Se ho del testo extra da appendere, lo aggiungo in coda ──────────────

    if input_data.system_message_content:
        system_message += (
            "\n\n## ADDITIONAL INSTRUCTIONS\n"
            "-----------------------------------------------------------------------------------------------------"
            f"{input_data.system_message_content}"
            "-----------------------------------------------------------------------------------------------------"
        )

    ####################################################################################################################

    # vector_store_config_id = f"{context}_vector_store_config"
    '''vectorstore_ids = [f"{context}_vector_store" for context in contexts]'''

    '''vectorstore_ids = []

    for ctx in contexts:
        # ricostruisci lo stesso base_cfg usato da _ensure_vector_store
        _base = {
                "vector_store_class": "Chroma",
                "params": {"persist_directory": f"vector_stores/{ctx}"},
                "embeddings_model_class": "OpenAIEmbeddings",
                "embeddings_params": {},  # valori non influenti per l'hash
                "description": f"Vector store for context {ctx}",
                "custom_metadata": {"source_context": ctx},
        }
        _, vs_id = make_vector_store_ids(_base)
        vectorstore_ids.append(vs_id)'''
    # ------------------------------------------------------------------
    # ● STEP 3 – assicuriamoci che ESISTA un VS per ogni context
    #            (nuovo ID se la api_key cambia)
    # ------------------------------------------------------------------
    vectorstore_ids: List[str] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for ctx in contexts:
            vs_id = await _ensure_vs_exists(ctx, llm_cfg.api_key, client)
            vectorstore_ids.append(vs_id)

    # Impostazione di configurazione per l'LLM basata su model_name (di default "gpt-4o")
    #llm_config_id = f"chat-openai_{model_name}_config"
    #llm_id = f"chat-openai_{model_name}"

    #async with httpx.AsyncClient() as client:
    #    # 1. Caricamento dell'LLM
    #    load_llm_url = f"{NLP_CORE_SERVICE}/llms/load_model/{llm_config_id}"
    #    llm_response = await client.post(load_llm_url, timeout=timeout_settings)
    #
    #    if llm_response.status_code != 200 and llm_response.status_code != 400:
    #        raise HTTPException(status_code=llm_response.status_code,
    #                            detail=f"Errore caricamento LLM: {llm_response.text}")
    #
    #    llm_load_result = llm_response.json()

    # vectorstore_ids = [vector_store_id]

    default_tools = [{"name": "VectorStoreTools", "kwargs": {"store_id": vectorstore_id}} for vectorstore_id in vectorstore_ids]

    default_tools.append({"name": "MongoDBTools",
                  "kwargs": {
                      "connection_string": "mongodb://localhost:27017",
                      "default_database": f"default_db",
                      "default_collection": "default_collection"
                  }})

    # ── Applico le custom_tools: sovrascrivo o aggiungo ────────────────────
    tools_by_name = {t["name"]: t for t in default_tools}

    for ct in input_data.custom_server_tools:
        tools_by_name[ct["name"]] = ct
    tools = list(tools_by_name.values())


    # Configurazione della chain
    chain_config = {
        "chain_type": "agent_with_tools",
        "config_id": f"{id_}_config", #_agent_with_tools_config",
        "chain_id": f"{id_}", #_agent_with_tools",
        "system_message": system_message, # #SYSTEM_MESSAGE,
        "llm_id": llm_id,  # Usa l'ID del modello LLM configurato
        "tools": tools,
        "extra_metadata": {
            "contexts": contexts,
            "model_name": model_name,
            "system_message_content": input_data.system_message_content,
            "custom_server_tools": input_data.custom_server_tools,
            #"client_tool_specs":  input_data.client_tool_specs
    }
    }

    async with httpx.AsyncClient() as client:
        try:
            # 1. Configura la chain
            configure_url = f"{NLP_CORE_SERVICE}/chains/configure_chain/"
            configure_response = await client.post(configure_url, json=chain_config)

            if configure_response.status_code != 200 and configure_response.status_code != 400:
                raise HTTPException(status_code=configure_response.status_code,
                                    detail=f"Errore configurazione: {configure_response.text}")

            configure_result = configure_response.json()

            # 2. Carica la chain
            load_url = f"{NLP_CORE_SERVICE}/chains/load_chain/{chain_config['config_id']}"
            load_response = await client.post(load_url)

            if load_response.status_code != 200 and load_response.status_code != 400:
                raise HTTPException(status_code=load_response.status_code,
                                    detail=f"Errore caricamento: {load_response.text}")

            load_result = load_response.json()

            return {
                "message": "Chain configurata e caricata con successo.",
                "llm_load_result": llm_load_result,
                "config_result": configure_result,
                "load_result": load_result
            }

        except httpx.HTTPStatusError as e:
            print(e)
            raise HTTPException(status_code=e.response.status_code, detail=f"Errore HTTP: {e.response.text}")

        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=f"Errore interno: {str(e)}")


# Retrieve metadata for a single context (by full path)
@app.get("/context_info/{context_path}", response_model=Dict[str, Any])
async def get_context_info(
    context_name: str,
    token: str | None = Query(None, description="Access‑token (facoltativo se REQUIRED_AUTH=False)")
):
    """
    Restituisce le informazioni del contesto indicato da context_path.
    Non crea nulla: cerca tra i context già esistenti.
    """
    # ──────────────────────────────────────────────────────────────────────────
    if REQUIRED_AUTH:
        verify_access_token(token, cognito_sdk)

    # Recupera l’elenco completo dei context dal core‑service
    all_ctx = await list_contexts_from_server()          # <- già definita sopra

    # Cerca il path esatto richiesto
    ctx_info = next((c for c in all_ctx if c["path"] == context_name), None)
    if ctx_info is None:
        raise HTTPException(status_code=404, detail="Context not found")

    return ctx_info



# Chain Execute API Interface
@app.post("/execute_chain", response_model=Dict[str, Any])
async def execute_chain(
        chain_id: str = Query(..., title="Chain ID", description="The unique ID of the chain to execute"),
        query: Dict[str, Any] = Body(..., example={"input": "What is my name?", "chat_history": []}),
        token: Optional[str] = Body(None)):

    if REQUIRED_AUTH:
        verify_access_token(token, cognito_sdk)

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{NLP_CORE_SERVICE}/chains/execute_chain/",
                                     json={"chain_id": chain_id, "query": query})
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.json())
        return response.json()


# Chain Stream API Interface
@app.post("/stream_chain")
async def stream_chain(
        chain_id: str = Query(..., title="Chain ID", description="The unique ID of the chain to stream"),
        query: Dict[str, Any] = Body(..., example={"input": "What is my name?", "chat_history": []}),
        token: str = Query(None, title="Token", description="Access token")):

    if REQUIRED_AUTH:
        verify_access_token(token, cognito_sdk)

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", f"{NLP_CORE_SERVICE}/chains/stream_chain/",
                                 json={"chain_id": chain_id, "query": query}) as response:
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.json())
            async for chunk in response.aiter_text():
                yield chunk


@app.get("/download", response_class=StreamingResponse)
async def download_file(
        file_id: str = Query(..., description="The ID or path of the file to download"),
        token: str = Query(None, description="Access Token")
):
    """
    Download a file by its ID via the intermediary API.
    """

    if REQUIRED_AUTH:
        verify_access_token(token, cognito_sdk)

    async with httpx.AsyncClient() as client:
        # Make a GET request to the source API's download endpoint
        response = await client.get(f"{NLP_CORE_SERVICE}/data_stores/download/{file_id}")
        if response.status_code != 200:
            # If the source API returns an error, raise an HTTPException
            raise HTTPException(status_code=response.status_code, detail=response.text)

        # Extract headers to forward them to the client
        headers = dict(response.headers)
        # Forward Content-Disposition and Content-Type headers
        content_type = headers.get('Content-Type', 'application/octet-stream')
        content_disposition = headers.get('Content-Disposition', f'attachment; filename="{file_id}"')

        # Return a StreamingResponse to stream the file content to the client
        return StreamingResponse(
            response.aiter_bytes(),
            media_type=content_type,
            headers={"Content-Disposition": content_disposition}
        )

class TaskStatusItem(BaseModel):
    task_id: str  # UUID del job
    kind: str #= Field(..., regex="^(loader|vector)$")

class TasksStatusRequest(BaseModel):
    tasks: List[TaskStatusItem] = Field(
        ...,
        description="Elenco dei task (loader o vector) da monitorare"
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Proxy per  **GET /document_stores/documents/{collection_name}/**
#  (stessa firma e stesso schema di risposta dell’API originale)
# ─────────────────────────────────────────────────────────────────────────────
     # ← aggiusta il path se diverso

@app.get("/documents/{collection_name}/") #), response_model=List[DocumentModel])
async def list_documents_proxy(
    collection_name: str = Path("", description="The name of the collection."),
    prefix: Optional[str] = Query(
        None, description="Prefix to filter documents (optional)."
    ),
    skip: int = Query(0, ge=0, description="Number of documents to skip."),
    limit: int = Query(10, ge=1, description="Maximum number of documents to return."),
    token: str = Query(None, description="Access Token")
):
    """
    Re-instrada la richiesta verso l’endpoint originale
    `GET /document_stores/documents/{collection_name}/
    e restituisce la stessa identica struttura dati.
    """

    if REQUIRED_AUTH:
        verify_access_token(token, cognito_sdk)

    params = {"skip": skip, "limit": limit}
    if prefix:
        params["prefix"] = prefix

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{NLP_CORE_SERVICE}/document_stores/documents/{collection_name}/",
            params=params,
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.json())

    # la risposta dell’API originale è già nel formato atteso da DocumentModel
    return resp.json()


async def _fetch_single_status(client: httpx.AsyncClient,
                               task_id: str,
                               kind: str) -> Dict[str, Any]:
    if kind == "loader":
        url = f"{NLP_CORE_SERVICE}/document_loaders/task_status/{task_id}"
    elif kind == "vector":
        url = f"{NLP_CORE_SERVICE}/vector_stores/vector_store/task_status/{task_id}"
    else:
        return {"status": "UNKNOWN_KIND"}

    r = await client.get(url)

    if r.status_code == 404:
        # task not yet registered – report PENDING instead of failing
        return {"status": "PENDING"}
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"Upstream error on {task_id}: {r.text}")

    return r.json()


@app.get("/tasks_status", response_model=Dict[str, Any])
async def get_tasks_status(
    tasks: List[str] = Query(...,
        description="Task-IDs separati da virgola. "
                    "Prefissa con loader: o vector: (es.: loader:uuid,vector:uuid)"),
    token: str = Query(None, description="Access Token")
):
    """
    Ritorna lo stato corrente di tutti i task richiesti.

    Esempio:
        /tasks_status?tasks=loader:1234,vector:abcd
    """

    if REQUIRED_AUTH:
        verify_access_token(token, cognito_sdk)

    task_items: List[TaskStatusItem] = []
    for raw in tasks[0].split(","):

        try:
            kind, tid = raw.split(":", 1)
            task_items.append(TaskStatusItem(task_id=tid, kind=kind))
        except ValueError:
            raise HTTPException(400, f"Formato task non valido: {raw}")

    async with httpx.AsyncClient() as client:
        coros = [
            _fetch_single_status(client, itm.task_id, itm.kind)
            for itm in task_items
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

    # costruiamo la risposta
    statuses: Dict[str, Any] = {}
    for itm, res in zip(task_items, results):
        if isinstance(res, Exception):
            statuses[itm.task_id] = {"status": "ERROR", "error": str(res)}
        else:
            statuses[itm.task_id] = res

    return {
        "requested": len(task_items),
        "statuses": statuses,
        "timestamp": datetime.utcnow().isoformat()
    }



#class ExecuteChainRequest(BaseModel):
#    chain_id: str = Field(..., example="example_chain", title="Chain ID", description="The unique ID of the chain to execute.")
#    query: Dict[str, Any] = Field(..., example={"input": "What is my name?", "chat_history": [["user", "hello, my name is mario!"], ["assistant", "hello, how are you mario?"]]}, title="Query", description="The input query for the chain.")
#    inference_kwargs: Dict[str, Any] = Field(..., example={}, description="")

class ExecuteChainRequest(BaseModel):
    token: Optional[str] = Field(..., description="Access token used to identify user")
    subscription_id: Optional[str] = Field(..., description="user's active subscription")
    chain_id: str = Field(..., description="The unique ID of the chain to execute.")
    # Legacy query (deprecato)
    query: Optional[Dict[str, Any]] = Field(
        None,
        description="(DEPRECATED) Legacy payload: {'input': text, 'chat_history': [...]}"
    )
    # Nuovi campi multimodali
    input_text: Optional[str] = Field(None, description="Testo del messaggio corrente")
    input_images: Optional[List[Dict[str, Any]]] = Field(
        None, description="Lista di URL/data-URI delle immagini del messaggio"
    )
    chat_history: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Cronologia: ogni item {'role':'user'/'assistant','parts':[...parts...]}"
    )
    inference_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parametri da passare a invoke/astream (es. {'stream':True})"
    )

    def get_payload(self) -> Dict[str, Any]:
        """
        Costruisce il payload da inoltrare al service:
        - se legacy `query` è presente → lo restituisce così com'è
        - altrimenti crea {'input': [...parts], 'chat_history': [...]}
        """
        if self.query:
            return {"chain_id": self.chain_id,
                    "query": self.query,
                    "inference_kwargs": self.inference_kwargs or {}}
        #parts = []
        #if self.input_text:
        #    parts.append({"type": "text", "text": self.input_text})
        #for url in (self.input_images or []):
        #    parts.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
        return {
            "chain_id": self.chain_id,
            "input_text": self.input_text,
            "input_images": self.input_images or [],
            "chat_history": self.chat_history or [],
            "inference_kwargs": self.inference_kwargs or {}
        }

#@app.post("/stream_events_chain")
async def stream_events_chain(
    body: ExecuteChainRequest,                  # lo stesso schema usato altrove
):
    """
    Proxy 1-to-1 di **POST /chains/stream_events_chain**:
    replica I/O byte-per-byte e mantiene lo stream invariato.
    """

    #if REQUIRED_AUTH:
    #    verify_access_token(token, cognito_sdk)

    # ------------------------------------------------------------------ #
    # Wrapper per rilanciare upstream e ributtare giù i chunk “as-is”.   #
    # ------------------------------------------------------------------ #
    async def relay():
        timeout = httpx.Timeout(600.0, connect=600.0, read=600.0, write=600.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{NLP_CORE_SERVICE}/chains/stream_events_chain",
                json=body.model_dump()        # stesso payload del servizio chains
            ) as resp:

                if resp.status_code != 200:
                    # Propaga l’errore così com’è
                    detail = await resp.aread()
                    raise HTTPException(resp.status_code, detail.decode())

                async for chunk in resp.aiter_bytes():
                    # **non** modifichiamo né ricomponiamo: passthrough puro
                    #print(chunk)
                    yield chunk

    # Il media_type è lo stesso dell’upstream (“application/json”)
    return StreamingResponse(relay(), media_type="application/json")


@app.post("/stream_events_chain")
async def stream_events_chain(
    body: ExecuteChainRequest,                  # lo stesso schema usato altrove
):
    """
    Proxy 1-to-1 di **POST /chains/stream_events_chain**:
    replica I/O byte-per-byte e mantiene lo stream invariato.
    """

    if REQUIRED_AUTH:
        verify_access_token(body.token, cognito_sdk)

    # === 1) ESTRAZIONE messaggio + history dal body ===
    msg = None
    hist = []
    if body.query:
        msg = (body.query or {}).get("input", "")
        hist = (body.query or {}).get("chat_history", []) or []
    else:
        msg = body.input_text or ""
        # body.chat_history è già nel formato nuovo [{'role':..., 'parts':[...]}, ...]
        # per la stima usiamo la lista di coppie semplice (user/assistant) se disponibile
        # fallback: serializza ruoli in una stringa flat
        try:
            hist = [[(h.get("role") or "user"), json.dumps(h.get("parts") or [])] for h in (body.chat_history or [])]
        except Exception:
            hist = []

    # === 2) STIMA COSTO ===
    icost = await estimate_chain_interaction_cost(
        EstimateInteractionRequest(
            chain_id=body.chain_id,
            chain_config=None,
            message=msg or "",
            chat_history=hist or [],
        )
    )
    credits_to_consume = icost.cost_total_usd

    # === 3) CONSUMO CREDITI ===
    # NB: qui il token non è passato come argomento esplicito;
    #     se vuoi richiederlo sempre, aggiungi 'token: Optional[str]' nei parametri

    await _consume_credits_or_402(
        body.token,
        credits_to_consume,
        reason=f"chat.stream_events chain_id={body.chain_id} len(message)={len(msg or '')}",
        subscription_id=body.subscription_id
    )

    # ------------------------------------------------------------------ #
    # Wrapper per rilanciare upstream e ributtare giù i chunk “as-is”.   #
    # ------------------------------------------------------------------ #
    async def relay():
        timeout = httpx.Timeout(600.0, connect=600.0, read=600.0, write=600.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
           payload = body.get_payload()
           print("🔔 proxy sending payload:", payload)
           async with client.stream(
                   "POST",
                   f"{NLP_CORE_SERVICE}/chains/stream_events_chain",
                   json=payload
           ) as resp:
                if resp.status_code != 200:
                    # Propaga l’errore così com’è
                    detail = await resp.aread()
                    raise HTTPException(resp.status_code, detail.decode())

                async for chunk in resp.aiter_bytes():
                    # **non** modifichiamo né ricomponiamo: passthrough puro
                    #print(chunk)
                    yield chunk

    # Il media_type è lo stesso dell’upstream (“application/json”)
    return StreamingResponse(relay(), media_type="application/json")


@app.get("/loaders_catalog", response_model=dict)
async def loaders_catalog():
    """
    Ritorna una mappa { <estensione_file>: [<loader1>, <loader2>, ...] }
    con i loader predefiniti disponibili per ciascuna estensione.
    """
    return {
        "png":  ["ImageDescriptionLoader"],
        "jpg":  ["ImageDescriptionLoader"],
        "jpeg": ["ImageDescriptionLoader"],
        "avi":  ["VideoDescriptionLoader", "VideoEventDetectionLoader"],
        "mp4":  ["VideoDescriptionLoader", "VideoEventDetectionLoader"],
        "mov":  ["VideoDescriptionLoader", "VideoEventDetectionLoader"],
        "mkv":  ["VideoDescriptionLoader", "VideoEventDetectionLoader"],
        "default": ["UnstructuredLoader"]
    }

@app.get("/loader_kwargs_schema", response_model=dict)
async def loader_kwargs_schema():
    """
    Ritorna lo schema dei parametri per ciascun loader.
    Ogni campo contiene: name, type, default, items (se enum), example ed editable.
    """
    return {
        "ImageDescriptionLoader": {
            "openai_api_key": {
                "name": "openai_api_key",
                "type": "string",
                "default": "<random-api-key>",
                "items": None,
                "example": "sk-abc123",
                "editable": False
            },
            "resize_to": {
                "name": "resize_to",
                "type": "tuple[int,int]",
                "default": [256, 256],
                "items": None,
                "example": [1024, 1024]
            }
        },
        "VideoDescriptionLoader": {
            "openai_api_key": {
                "name": "openai_api_key",
                "type": "string",
                "default": "<random-api-key>",
                "items": None,
                "example": "sk-xyz789",
                "editable": False
            },
            "resize_to": {
                "name": "resize_to",
                "type": "list[int,int]",
                "default": [256, 256],
                "items": None,
                "example": [1024, 1024]
            },
            "num_frames": {
                "name": "num_frames",
                "type": "int",
                "default": 10,
                "items": None,
                "example": 20
            },
            "frame_rate": {
                "name": "frame_rate",
                "type": "int",
                "default": None,
                "items": None,
                "example": 2
            }
        },
        "VideoEventDetectionLoader": {
            "openai_api_key": {
                "name": "openai_api_key",
                "type": "string",
                "default": "<random-api-key>",
                "items": None,
                "example": "sk-evt000",
                "editable": False
            },
            "resize_to": {
                "name": "resize_to",
                "type": "list[int,int]",
                "default": [256, 256],
                "items": None,
                "example": [640, 640]
            },
            "frame_rate": {
                "name": "frame_rate",
                "type": "int",
                "default": None,
                "items": None,
                "example": 1
            },
            "num_frames": {
                "name": "num_frames",
                "type": "int",
                "default": None,
                "items": None,
                "example": 500
            },
            "window_size_seconds": {
                "name": "window_size_seconds",
                "type": "int",
                "default": 10,
                "items": None,
                "example": 3
            },
            "window_overlap_seconds": {
                "name": "window_overlap_seconds",
                "type": "int",
                "default": 2,
                "items": None,
                "example": 1
            },
            "batch_size": {
                "name": "batch_size",
                "type": "int",
                "default": 4,
                "items": None,
                "example": 20
            },
            "max_concurrency": {
                "name": "max_concurrency",
                "type": "int",
                "default": 10,
                "items": None,
                "example": 20
            },
            "event_prompt": {
                "name": "event_prompt",
                "type": "string",
                "default": "",
                "items": None,
                "example": "Rileva tutte le volte in cui compare una macchina."
            }
        },
        "UnstructuredLoader": {
            # --- instradamento verso API self-hosted ---
            "partition_via_api": {
                "name": "partition_via_api",
                "type": "boolean",
                "default": True,
                "items": None,
                "example": True,
                "editable": True
            },
            "url": {
                "name": "url",
                "type": "string",
                "default": "http://34.13.153.241:8333/",
                "items": None,
                "example": "http://34.13.153.241:8333/",
                "editable": True
            },
            "api_key": {
                "name": "api_key",
                "type": "string",
                "default": "metti-una-chiave-robusta", #"<set-in-env>",
                "items": None,
                "example": "metti-una-chiave-robusta",
                "editable": True
            },

            # --- modalità di ritorno documenti dal loader ---
            "mode": {
                "name": "mode",
                "type": "string",
                "default": "paged",
                "items": ["single", "elements", "paged"],
                "example": "elements"
            },

            # --- strategia / modello / formato ---
            "strategy": {
                "name": "strategy",
                "type": "string",
                "default": "hi_res",
                "items": ["fast", "hi_res", "ocr_only"], #"auto"
                "example": "hi_res"
            },
            "hi_res_model_name": {
                "name": "hi_res_model_name",
                "type": "string",
                "default": "yolox",
                "items": ["yolox", "detectron2_onnx"],
                "example": "yolox"
            },
            "output_format": {
                "name": "output_format",
                "type": "string",
                "default": "application/json",
                "items": ["application/json", "text/csv"],
                "example": "application/json"
            },

            # --- OCR / lingue / encoding ---
            "ocr_languages": {
                "name": "ocr_languages",
                "type": "list[string]",
                "default": ["ita", "eng"],
                "items": None,
                "example": ["ita", "eng"]
            },
            "languages": {
                "name": "languages",
                "type": "list[string]",
                "default": ["it", "en"],
                "items": None,
                "example": ["it", "en"]
            },
            "encoding": {
                "name": "encoding",
                "type": "string",
                "default": "utf-8",
                "items": None,
                "example": "utf-8"
            },

            # --- layout / coordinate / pagine / slide ---
            "coordinates": {
                "name": "coordinates",
                "type": "boolean",
                "default": False,
                "items": None,
                "example": True
            },
            "include_page_breaks": {
                "name": "include_page_breaks",
                "type": "boolean",
                "default": False,
                "items": None,
                "example": True
            },
            "starting_page_number": {
                "name": "starting_page_number",
                "type": "integer",
                "default": 1,
                "items": None,
                "example": 1
            },
            "include_slide_notes": {
                "name": "include_slide_notes",
                "type": "boolean",
                "default": True,
                "items": None,
                "example": True
            },

            # --- tabelle PDF / XML ---
            "pdf_infer_table_structure": {
                "name": "pdf_infer_table_structure",
                "type": "boolean",
                "default": True,
                "items": None,
                "example": True
            },
            "skip_infer_table_types": {
                "name": "skip_infer_table_types",
                "type": "list[string]",
                "default": [],
                "items": None,
                "example": ["pdf"]
            },
            "xml_keep_tags": {
                "name": "xml_keep_tags",
                "type": "boolean",
                "default": False,
                "items": None,
                "example": False
            },

            # --- immagini estratte (opzionale, dipende dalla tua pipeline) ---
            "extract_image_block_types": {
                "name": "extract_image_block_types",
                "type": "list[string]",
                "default": [],
                "items": None,
                "example": ["table", "figure"]
            },
            "unique_element_ids": {
                "name": "unique_element_ids",
                "type": "boolean",
                "default": True,
                "items": None,
                "example": True
            },

            # --- chunking lato Unstructured ---
            "chunking_strategy": {
                "name": "chunking_strategy",
                "type": "string",
                "default": "by_title",
                "items": ["basic", "by_title"],
                "example": "by_title"
            },
            "combine_under_n_chars": {
                "name": "combine_under_n_chars",
                "type": "integer",
                "default": 2000,
                "items": None,
                "example": 2000
            },
            "max_characters": {
                "name": "max_characters",
                "type": "integer",
                "default": 4000,
                "items": None,
                "example": 4000
            },
            "multipage_sections": {
                "name": "multipage_sections",
                "type": "boolean",
                "default": True,
                "items": None,
                "example": True
            },
            "new_after_n_chars": {
                "name": "new_after_n_chars",
                "type": "integer",
                "default": None,
                "items": None,
                "example": 1500
            },
            "overlap": {
                "name": "overlap",
                "type": "integer",
                "default": 200,
                "items": None,
                "example": 200
            },
            "overlap_all": {
                "name": "overlap_all",
                "type": "boolean",
                "default": False,
                "items": None,
                "example": False
            },

            # --- lato client HTTP (opzionale) ---
            "request_timeout_seconds": {
                "name": "request_timeout_seconds",
                "type": "integer",
                "default": 180,
                "items": None,
                "example": 300
            },
            "retries": {
                "name": "retries",
                "type": "integer",
                "default": 2,
                "items": None,
                "example": 3
            }
        }
    }

'''"UnstructuredLoader": {
  "mode": {
    "name": "mode",
    "type": "string",
    "default": "single",
    "items": ["single", "elements"],
    "example": "elements",
    #"description": "– “single”: ritorna un unico Document con tutto il contenuto.  \n– “elements”: un Document per ogni elemento estratto (paragrafi, titoli, immagini, PageBreak…)."
  },
  "chunking_strategy": {
    "name": "chunking_strategy",
    "type": "string",
    "default": "basic",
    "items": ["basic", "by_title"], #, "by_page", "by_similarity"],
    "example": "basic",
    #"description": "Strategia di suddivisione in chunk:  \n– basic: fill fino ai limiti di caratteri  \n– by_title: split su ogni Title  \n– by_page: un chunk = una pagina (tramite API)  \n– by_similarity: group topic‑wise (tramite API)"
  },
            "strategy": {
                "name": "strategy",
                "type": "string",
                "default": "auto",
                "items": ["auto", "fast", "hi_res", "ocr_only"],
                "example": "fast",
                #"description": "Modalità di parsing per PDF/immagini: “fast” per velocità, “hi_res” per layout di precisione, “ocr_only” per solo OCR."
            },
  "max_characters": {
    "name": "max_characters",
    "type": "integer",
    "default": 500,
    "example": 1500,
    #"description": "Hard cap sul numero massimo di caratteri per chunk."
  },
  "new_after_n_chars": {
    "name": "new_after_n_chars",
    "type": "integer",
    "default": 500,
    "example": 1000,
    #"description": "Soft cap: suggerisce il punto di break, fino a un massimo di max_characters."
  },
  "overlap": {
    "name": "overlap",
    "type": "integer",
    "default": 0,
    "example": 200,
    #"description": "Numero di caratteri ripresi all’inizio di ogni nuovo chunk."
  },
  "overlap_all": {
    "name": "overlap_all",
    "type": "boolean",
    "default": False,
    "example": False,
    #"description": "Se true applica overlap anche fra chunk che non superano max_characters."
  },
  "include_page_breaks": {
    "name": "include_page_breaks",
    "type": "boolean",
    "default": False,
    "example": True,
    #"description": "Inietta elementi PageBreak nei risultati per mantenere i confini di pagina."
  },
            "partition_via_api": {
                "name": "partition_via_api",
                "type": "boolean",
                "default": False,
                "example": False,
                #"description": "Se true forza l’uso del Partition Endpoint remoto invece del parsing locale."
            },
}
    }'''










# ─────────────────────────────────────────────────────────────────────────────
# COST‑ESTIMATE ENDPOINT – formula, params, params_conditions
# ─────────────────────────────────────────────────────────────────────────────
HIRES_PRICE_PER_PAGE  = float(os.getenv("HIRES_PRICE_PER_PAGE",  "0.01")) * 1000  # USD
FAST_PRICE_PER_PAGE   = float(os.getenv("FAST_PRICE_PER_PAGE",   "0.001")) * 1000
IMAGE_FLAT_COST_USD   = float(os.getenv("IMAGE_FLAT_COST_USD",   "0.005")) * 1000  # USD / img
VIDEO_PRICE_PER_MIN   = float(os.getenv("VIDEO_PRICE_PER_MIN",   "0.10"))  * 1000 # USD / min
FALLBACK_KB_PER_PAGE  = 100                                                # ↳ csv / txt …
TOKENS_PER_PAGE = 1000          # ≈ 1k‑token ≃ 4000 caratteri

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
PAGE_DOCS = {".pdf", ".pptx", ".docx", ".tif", ".tiff"}

# --- formati testuali “puri” da cui estraiamo direttamente testo ---
TEXT_EXT = {
    ".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".html", ".htm", ".xml"
}

# ── Pydantic ────────────────────────────────────────────────────────────────
class FileCost(BaseModel):
    filename        : str
    kind            : str                   # document | image | video
    pages           : int    | None = None
    minutes         : float  | None = None
    strategy        : str    | None = None
    size_bytes      : int    | None = None
    tokens_est      : int    | None = None
    cost_usd        : float  | None = None
    formula         : str    | None = None
    params          : Dict[str, Any] | None = None
    params_conditions: Dict[str, str] | None = None   # 👈 NUOVO
    error           : str    | None = None

class CostEstimateResponse(BaseModel):
    files       : List[FileCost]
    grand_total : float

# ── Helpers ─────────────────────────────────────────────────────────────────
def _price_per_page(strategy: str) -> float:
    return HIRES_PRICE_PER_PAGE if strategy == "hi_res" else FAST_PRICE_PER_PAGE

def _estimate_text_pages(blob: bytes) -> int:
    """
    Conta i token (≈ len(text)/4) e li converte in pagine.
    """
    try:
        text = blob.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        text = blob.decode("latin-1", errors="ignore")

    tokens = math.ceil(len(text) / 4)
    print(text, tokens)# 1 token ≈ 4 char
    return max(1, math.ceil(tokens / TOKENS_PER_PAGE))

# ─────────────────────────────────────────────────────────────────────────────
# Helper: pagine documento
# ─────────────────────────────────────────────────────────────────────────────
def _estimate_pages(ext: str, content: bytes) -> int:
    ext = ext.lower()
    bio = BytesIO(content)

    if ext == ".pdf":
        return len(PdfReader(bio).pages)

    elif ext == ".pptx":
        return len(Presentation(bio).slides)

    elif ext == ".docx":
        doc = docx.Document(bio)
        words = sum(len(p.text.split()) for p in doc.paragraphs)
        return max(1, math.ceil(words / 800))  # 800 parole ≈ 1 pagina

    elif ext in {".tif", ".tiff"}:
        img = Image.open(bio)
        return getattr(img, "n_frames", 1)  # multipage‑TIFF

    # Fallback: 100 KB → 1 pagina (regola Unstructured)
    return math.ceil(len(content) / 102_400)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: durata video in minuti (usa moviepy se disponibile, altrimenti size‑heuristic)
# ─────────────────────────────────────────────────────────────────────────────
def _estimate_video_minutes(tmp_path: Path) -> float:
    try:
        from moviepy import VideoFileClip          # lazy‑import
        with VideoFileClip(tmp_path) as clip:
            return clip.duration / 60.0
    except Exception:
        # Heuristica: 5 MB ≈ 1 minuto
        size_mb = tmp_path.stat().st_size / (1024 * 1024)
        return size_mb / 5.0                              # molto approssimativo

# prima era:  def _choose_strategy(kwargs: dict | None) -> str:
def _choose_strategy(
    kwargs: dict | None,
    ext: str,
    size_bytes: int
) -> str:
    """
    Ritorna la strategy da usare.
    -   'hi_res' / 'fast'  → pass-through
    -   'auto'             → heuristic “stile Unstructured”:
            • se (kind==image) OR (size_bytes < 200 KB) → fast
            • altrimenti                               → hi_res
    """

    raw = (kwargs.get(ext[1:] if ext.startswith(".") else ext, {}) or {}).get("strategy", "hi_res")

    if raw != "auto":
        return raw            # 'hi_res' o 'fast' espliciti

    # --- heuristica per la modalità AUTO -------------------------
    is_img = ext in IMAGE_EXT
    if is_img or size_bytes < 200_000:        # <≈200 KB
        return "fast"
    return "hi_res"

# ── Endpoint ────────────────────────────────────────────────────────────────
@app.post("/estimate_file_processing_cost", response_model=CostEstimateResponse)
async def estimate_file_processing_cost(
    files        : List[UploadFile] = File(...),
    #loaders      : str | None = Form(None),
    loader_kwargs: str | None = Form(None),
):


    try:
        kwargs_map  = json.loads(loader_kwargs) if loader_kwargs else {}
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"loader_kwargs JSON non valido: {e}")

    results: List[FileCost] = []

    # ① prima immagini & video, poi documenti
    ordered = sorted(
        files,
        key=lambda f: 0 if Path(f.filename).suffix.lower() in IMAGE_EXT
                         or Path(f.filename).suffix.lower() in VIDEO_EXT
                      else 1
    )

    for up in ordered:
        ext  = Path(up.filename).suffix.lower()
        kind = ("image" if ext in IMAGE_EXT else
                "video" if ext in VIDEO_EXT else
                "document")

        try:
            blob   = await up.read()
            size_b = len(blob)

            # ───── VIDEO ──────────────────────────────────────────────────
            if kind == "video":
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(blob)
                    tmp_path = Path(tmp.name)

                try:
                    minutes = _estimate_video_minutes(tmp_path)
                finally:
                    tmp_path.unlink(missing_ok=True)

                results.append(FileCost(
                    filename = up.filename,
                    kind     = kind,
                    minutes  = round(minutes, 2),
                    size_bytes = size_b,
                    cost_usd = round(minutes * VIDEO_PRICE_PER_MIN, 4),
                    formula  = "cost = {minutes} * {VIDEO_PRICE_PER_MIN}",
                    params   = {
                        "minutes"            : round(minutes, 2),
                        "VIDEO_PRICE_PER_MIN": VIDEO_PRICE_PER_MIN,
                    },
                    params_conditions = {}          # nessuna condizione
                ))
                continue

            # ───── IMMAGINI ───────────────────────────────────────────────
            if kind == "image":
                results.append(FileCost(
                    filename   = up.filename,
                    kind       = kind,
                    size_bytes = size_b,
                    cost_usd   = IMAGE_FLAT_COST_USD,
                    formula    = "cost = {IMAGE_FLAT_COST_USD}",
                    params     = {"IMAGE_FLAT_COST_USD": IMAGE_FLAT_COST_USD},
                    params_conditions = {}
                ))
                continue

            # ───── DOCUMENTI  ────────────────────────────────────────────
            strategy = _choose_strategy(kwargs_map, ext, size_b)

            price_page = _price_per_page(strategy)

            base_params = {
                # valori possibili *indipendenti* dalla scelta
                "HIRES_PRICE_PER_PAGE": HIRES_PRICE_PER_PAGE,
                "FAST_PRICE_PER_PAGE" : FAST_PRICE_PER_PAGE,
                "strategy"            : None,
            }

            cond = {
                "price_per_page": "{HIRES_PRICE_PER_PAGE} if {strategy}=='hi_res' "
                                  "else {FAST_PRICE_PER_PAGE}"
            }

            if ext in PAGE_DOCS:
                pages = _estimate_pages(ext, blob)
                results.append(FileCost(
                    filename   = up.filename,
                    kind       = kind,
                    pages      = pages,
                    #strategy   = strategy,
                    size_bytes = size_b,
                    tokens_est = None, #round(min(size_b, 500_000) / 4),
                    cost_usd   = round(pages * price_page, 4),
                    formula    = "cost = {pages} * {price_per_page}",
                    params     = base_params | {  # unione dizionari (3.9+)
                        "pages"         : pages,
                        "price_per_page": None,
                        "strategy": None
                    },
                    params_conditions = cond,
                ))

                continue

            # --- Documenti TESTUALI PURI  ---------------------------------
            elif ext in TEXT_EXT:
                pages_est = _estimate_text_pages(blob)
                tokens    = pages_est * TOKENS_PER_PAGE
                results.append(FileCost(
                    filename   = up.filename,
                    kind       = kind,
                    pages      = pages_est,
                    #strategy   = strategy,
                    size_bytes = size_b,
                    cost_usd   = round(pages_est * price_page, 4),
                    formula    = (
                        "cost = {pages_est} * {price_per_page}"
                    ),
                    params     = base_params | {
                        "tokens"         : tokens,
                        "TOKENS_PER_PAGE": TOKENS_PER_PAGE,
                        "pages_est": pages_est,
                        "price_per_page" : None,
                        "strategy": None,
                    },
                    params_conditions = cond,
                ))
                continue   #  <‑‑ per saltare al prossimo file

            else:
                pages_est = math.ceil(size_b / (FALLBACK_KB_PER_PAGE * 1024))
                results.append(FileCost(
                    filename   = up.filename,
                    kind       = kind,
                    pages      = pages_est,
                    #strategy   = strategy,
                    size_bytes = size_b,
                    cost_usd   = round(pages_est * price_page, 4),
                    formula    = ("cost = ceil({size_bytes} / ({KB_per_page_rule} * 1024)) * {price_per_page}"),
                    params     = base_params | {
                        "size_bytes"     : size_b,
                        "pages_est"      : pages_est,
                        "KB_per_page_rule": FALLBACK_KB_PER_PAGE,
                        "price_per_page" : None,
                        "strategy": None
                    },
                    params_conditions = cond
                ))

        except Exception as exc:
            results.append(FileCost(
                filename = up.filename,
                kind     = kind,
                error    = str(exc),
            ))

    grand_total = round(sum(f.cost_usd or 0.0 for f in results), 4)

    return CostEstimateResponse(files=results, grand_total=grand_total)






# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: stima costo di UNA interazione con una chain/agent
# ─────────────────────────────────────────────────────────────────────────────
import os, math, json, httpx
from typing import List, Dict, Any
from fastapi import HTTPException, Body
from pydantic import BaseModel, Field

# ╭──── prezzi (override via env) ───────────────────────────────────────────╮
GPT4O_IN_PRICE        = float(os.getenv("GPT4O_IN_PRICE",        "0.01")) * 1000  # USD / 1k tok
GPT4O_OUT_PRICE       = float(os.getenv("GPT4O_OUT_PRICE",       "0.03")) * 1000
GPT4O_MINI_IN_PRICE   = float(os.getenv("GPT4O_MINI_IN_PRICE",   "0.002")) * 1000
GPT4O_MINI_OUT_PRICE  = float(os.getenv("GPT4O_MINI_OUT_PRICE",  "0.006")) * 1000
# ╰──────────────────────────────────────────────────────────────────────────╯
PER_TOOL_TOKEN_EST    = int(os.getenv("PER_TOOL_TOKEN_EST",      "300"))
DEFAULT_TOOLS_COUNT   = int(os.getenv("DEFAULT_TOOLS_COUNT",     "3"))
MAX_OUTPUT_TOKENS_DEF = int(os.getenv("MAX_OUTPUT_TOKENS",       "500"))

# ── INPUT / OUTPUT models ──────────────────────────────────────────────────
class EstimateInteractionRequest(BaseModel):
    chain_id     : str | None = None
    chain_config : Dict[str, Any] | None = None
    message      : str
    chat_history : List[List[str]] = Field(default_factory=list)

class InteractionCost(BaseModel):
    model_name      : str
    input_tokens    : int
    output_tokens   : int
    total_tokens    : int
    cost_input_usd  : float
    cost_output_usd : float
    cost_total_usd  : float
    formula         : str
    params          : Dict[str, Any]
    params_conditions: Dict[str, str]

# ── helpers ----------------------------------------------------------------
def _tok_est(text: str) -> int:          # 1 token ≈ 4 caratteri
    return math.ceil(len(text) / 4)

def _price_for(model_name: str) -> tuple[float, float, str, str]:
    """
    → (price_in, price_out, cond_in_str, cond_out_str)
    """
    cond_in  = ("{GPT4O_MINI_IN_PRICE}  if {model_name} = 'gpt-4o-mini' "
                "else {GPT4O_IN_PRICE}")
    cond_out = ("{GPT4O_MINI_OUT_PRICE} if {model_name} = 'gpt-4o' "
                "else {GPT4O_OUT_PRICE}")

    if "mini" in model_name.lower():
        return GPT4O_MINI_IN_PRICE, GPT4O_MINI_OUT_PRICE, cond_in, cond_out
    return GPT4O_IN_PRICE, GPT4O_OUT_PRICE, cond_in, cond_out

async def _get_chain_config(cfg_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{NLP_CORE_SERVICE}/chains/chain_config/{cfg_id}")
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"Chain‑config '{cfg_id}' non trovata")
    return r.json()

async def _get_llm_config(llm_id: str) -> Dict[str, Any] | None:
    cfg_id = f"{llm_id}_config" if not llm_id.endswith("_config") else llm_id
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{NLP_CORE_SERVICE}/llms/configuration/{cfg_id}")
    return r.json() if r.status_code == 200 else None


# ── Endpoint ----------------------------------------------------------------
@app.post("/estimate_chain_interaction_cost", response_model=InteractionCost)
async def estimate_chain_interaction_cost(
    body: EstimateInteractionRequest = Body(...)
):
    # 0️⃣  Recupero configurazione chain ----------------------------------------
    if body.chain_config:
        chain_cfg = body.chain_config
    elif body.chain_id:
        cfg_id   = body.chain_id if body.chain_id.endswith("_config") \
                                else f"{body.chain_id}_config"
        chain_cfg = await _get_chain_config(cfg_id)
    else:
        raise HTTPException(422, "Serve chain_id o chain_config")

    # 1️⃣  Elementi principali della chain --------------------------------------
    system_msg = chain_cfg.get("system_message", "")
    tools      = chain_cfg.get("tools", [])
    llm_id     = chain_cfg.get("llm_id", "")

    # 2️⃣  Config LLM  →  modello & max‑tokens out ------------------------------
    llm_cfg        = await _get_llm_config(llm_id)
    model_name     = (
        llm_cfg.get("model_kwargs", {}).get("model_name")
        if llm_cfg else "gpt-4o"
    )
    max_out_tokens = (
        llm_cfg.get("model_kwargs", {}).get("max_tokens", MAX_OUTPUT_TOKENS_DEF)
        if llm_cfg else MAX_OUTPUT_TOKENS_DEF
    )

    # 3️⃣  StifetchInitialCostma token *distinta* per ogni sorgente -----------------------------
    tokens_system   = _tok_est(system_msg)
    tokens_user     = _tok_est(body.message)
    tokens_history  = _tok_est(" ".join(m for _, m in body.chat_history))
    tokens_tools    = (len(tools) or DEFAULT_TOOLS_COUNT) * PER_TOOL_TOKEN_EST

    # input totale
    input_tokens = tokens_system + tokens_user + tokens_history + tokens_tools
    output_tokens = 500 #max_out_tokens      # ↖️ puoi cambiarlo se vuoi fisso

    # 4️⃣  Prezzi & costi --------------------------------------------------------
    price_in, price_out, cond_in, cond_out = _price_for(model_name)

    cost_in  = round(input_tokens  / 1_000 * price_in , 4)
    cost_out = round(output_tokens / 1_000 * price_out, 4)
    total    = round(cost_in + cost_out, 4)

    # 5️⃣  Formula esplicita -----------------------------------------------------
    formula = (
        "cost_total = (({tokens_system} + {tokens_user} + "
        "{tokens_history} + {tokens_tools}) / 1000) * {price_in} "
        "+ ({output_tokens} / 1000) * {price_out}"
    )

    # 6️⃣  Params dettagliati (per UI & ricalcoli locali) -----------------------
    params = {
        "tokens_system"  : tokens_system,
        "tokens_user"    : tokens_user,
        "tokens_history" : tokens_history,
        "output_tokens"  : output_tokens,
        "price_in"       : price_in,
        "price_out"      : price_out,
        "model_name"     : None,          # viene risolto lato client (params_conditions)
    }

    # 7️⃣  Condizioni per i prezzi (restano uguali) -----------------------------
    params_conditions = {
        "price_in" : cond_in,
        "price_out": cond_out,
    }

    # 8️⃣  Response -------------------------------------------------------------
    return InteractionCost(
        model_name        = model_name,
        input_tokens      = input_tokens,
        output_tokens     = output_tokens,
        total_tokens      = input_tokens + output_tokens,
        cost_input_usd    = cost_in,
        cost_output_usd   = cost_out,
        cost_total_usd    = total,
        formula           = formula,
        params            = params,
        params_conditions = params_conditions,
    )


@app.post("/get_chain_configuration", response_model=Dict[str, Any])
async def get_chain_configuration(body: GetChainConfigurationRequest):
    """
    Restituisce la configurazione di una chain e garantisce
    che il campo `contexts` sia presente (ricavato da extra_metadata
    o – in fallback – dai VectorStoreTools).
    """
    # ── autenticazione (se serve) ──────────────────────────────────────────
    if REQUIRED_AUTH:
        verify_access_token(body.token, cognito_sdk)

    # ── quale config_id devo chiedere? ------------------------------------
    if body.chain_id:
        config_id = (
            body.chain_id if body.chain_id.endswith("_config")
            else f"{body.chain_id}_config"
        )
    elif body.chain_config_id:
        config_id = (
            body.chain_config_id if body.chain_config_id.endswith("_config")
            else f"{body.chain_config_id}_config"
        )
    else:
        raise HTTPException(422, "Devi fornire chain_id oppure chain_config_id")

    # ── chiamata upstream ---------------------------------------------------
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{NLP_CORE_SERVICE}/chains/chain_config/{config_id}")

    if resp.status_code != 200:
        # propaghiamo l'errore originale
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise HTTPException(resp.status_code, detail)

    cfg = resp.json()

    # ── normalizziamo il nuovo campo `contexts` -----------------------------
    # 1) se presente in extra_metadata
    extra_meta = cfg.get("extra_metadata") or {}
    contexts = extra_meta.get("contexts")
    model_name = extra_meta.get("model_name")

    # 2) fallback: deducilo dai VectorStoreTools (store_id ➜ potrebbe essere hash)
    if contexts is None:
        contexts = [
            t["kwargs"]["store_id"]
            for t in cfg.get("tools", [])
            if t.get("name") == "VectorStoreTools"
        ]

    cfg["contexts"] = contexts
    return cfg









class ImageBase64Response(BaseModel):
    url: str
    content_type: str
    size_bytes: int
    width: int | None = None
    height: int | None = None
    sha256: str | None = None
    base64_raw: str          # solo i caratteri base64
    data_uri: str            # "data:<content-type>;base64,<...>"


async def _fetch_image_bytes(url: str, *, timeout_s: float = 20.0, max_bytes: int = 10 * 1024 * 1024) -> tuple[bytes, str]:
    """
    Scarica i byte dell'immagine via httpx e ritorna (bytes, content_type).
    Valida dimensione massima e content-type (deve iniziare con 'image/').
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
        r = await client.get(url, headers={"Accept": "image/*"})
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"Impossibile scaricare l'immagine: {r.text}")

        ctype = r.headers.get("Content-Type", "")
        # Nota: alcuni server non settano correttamente il content-type.
        # In quel caso tenteremo una detection via PIL più avanti.
        content = await r.aread()

        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Immagine troppo grande: {len(content)} bytes (limite {max_bytes})"
            )

        return content, ctype


# image_base64.py
import base64
import hashlib
import io
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from PIL import Image
from pydantic import BaseModel, Field

router = APIRouter()


# -----------------------------  MODELLI  ----------------------------- #
class ImageBase64Response(BaseModel):
    url: str
    content_type: str
    size_bytes: int
    sha256: str
    # opzionali: possono valere None se non leggiamo i metadati
    width: Optional[int] = Field(None, ge=1)
    height: Optional[int] = Field(None, ge=1)
    # i due campi sotto sono mutualmente esclusivi: tienili entrambi se ti servono
    base64_raw: str
    data_uri: str


# -----------------------------  HELPERS  ----------------------------- #
async def fetch_image_bytes(
    url: str,
    *,
    max_bytes: int,
    timeout_s: float = 10.0,
) -> tuple[bytes, str]:
    """
    Scarica l'immagine a blocchi senza superare `max_bytes`.
    Restituisce (raw_bytes, content_type header o stringa vuota).
    """
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s)
    ) as client, client.stream("GET", url, follow_redirects=True) as resp:
        if resp.status_code >= 400:
            raise HTTPException(resp.status_code, f"Errore HTTP {resp.status_code}")

        ctype = resp.headers.get("Content-Type", "").lower()

        # lettura a blocchi   (streaming=True evita di caricare tutto in memoria se gigante)
        buf = io.BytesIO()
        async for chunk in resp.aiter_bytes():
            buf.write(chunk)
            if buf.tell() > max_bytes:
                raise HTTPException(
                    413, f"File troppo grande (> {max_bytes // 1024} kB)"
                )

        return buf.getvalue(), ctype


# -----------------------------  ENDPOINT  ---------------------------- #
@app.get("/image/base64", response_model=ImageBase64Response)
async def image_to_base64(
    url: str = Query(..., description="URL assoluto dell'immagine"),
    include_dimensions: bool = Query(
        True, description="Se true prova a leggere width/height con Pillow"
    ),
    max_bytes: int = Query(
        10 * 1024 * 1024, description="Limite massimo in bytes che accettiamo"
    ),
):
    """
    Converte l’immagine remota in stringa Base64 (più data-URI) restituendo
    anche dimensioni (px) e SHA-256.
    """
    try:
        raw, ctype = await fetch_image_bytes(url, max_bytes=max_bytes)

        # ---------------- hash e dimensioni ---------------- #

        try:
            sha = hashlib.sha256(raw).hexdigest()
        except Exception as e:
            sha = None

        width = height = None

        if include_dimensions:
            try:
                with Image.open(io.BytesIO(raw)) as img:
                    width, height = img.size

                    # Se il content-type HTTP era assente/errato, inferiscilo da Pillow
                    if not ctype.startswith("image/"):
                        fmt = (img.format or "jpeg").lower()
                        ctype = f"image/{fmt}"
            except Exception:
                # niente crash: semplicemente lasciamo width/height = None
                pass

        # ---------------- normalizzazione content-type ------ #
        if not ctype.startswith("image/"):
            ctype = "image/octet-stream"

        # ---------------- encoding Base64 ------------------- #
        b64_raw = base64.b64encode(raw).decode("ascii")
        data_uri = f"data:{ctype};base64,{b64_raw}"

        return ImageBase64Response(
            url=url,
            content_type=ctype,
            size_bytes=len(raw),
            sha256=sha,
            width=width,
            height=height,
            base64_raw=b64_raw,
            data_uri=data_uri,
        )

    except HTTPException:
        raise  # propaghiamo così FastAPI mantiene il codice corretto
    except Exception as exc:
        # Log di debug lato server, ma maschera il messaggio verso il client
        print(f"[image_to_base64] errore non gestito: {exc}")
        raise HTTPException(500, "Errore interno in image_to_base64")



# --- [NEW] Schemi input/output endpoint Payments ----------------------------
class CurrentPlanResponse(BaseModel):
    subscription_id: str
    status: str | None = None
    plan_type: str | None = None
    variant: str | None = None
    pricing_method: str | None = None
    active_price_id: str | None = None
    period_start: int | None = None
    period_end: int | None = None

class CheckoutVariantIn(BaseModel):
    token: str | None = None
    plan_type: str
    variant: str
    locale: str | None = "it"
    success_url: str | None = None
    cancel_url: str | None = None

# --- [NEW] hint dal client per evitare round-trip di stato -------------------
class DeeplinkUpgradeIn(BaseModel):
    token: Optional[str] = None
    return_url: Optional[str] = None

    # target: EITHER target_price_id OR (target_plan_type + target_variant)
    target_price_id: Optional[str] = None
    target_plan_type: Optional[str] = None
    target_variant: Optional[str] = None

    # Hints per saltare letture
    current_subscription_id: Optional[str] = None
    current_plan_type: Optional[str] = None
    current_variant: Optional[str] = None

    @model_validator(mode="after")
    def _validate_target(self):
        if not self.target_price_id and not (self.target_plan_type and self.target_variant):
            raise ValueError("Devi fornire target_price_id oppure target_plan_type + target_variant.")
        return self


class PortalSessionIn(BaseModel):
    token: Optional[str] = None
    return_url: Optional[str] = None
    # hint
    current_subscription_id: Optional[str] = None
    current_plan_type: Optional[str] = None


class DeeplinkUpdateIn(BaseModel):
    token: Optional[str] = None
    return_url: Optional[str] = None
    change_intent: Optional[ChangeIntent] = Field(default=ChangeIntent.both)
    variants_override: Optional[List[str]] = None
    variants_catalog: Optional[List[str]] = None
    # hint
    current_subscription_id: Optional[str] = None
    current_plan_type: Optional[str] = None
    current_variant: Optional[str] = None


class DeeplinkCancelIn(BaseModel):
    token: Optional[str] = None
    return_url: Optional[str] = None
    immediate: bool = True
    # facoltativo: utile solo se il backend richiede anche un preset
    portal_preset: Optional[str] = None
    # per soddisfare il requisito server "portal_preset o variants_override"
    variants_catalog: Optional[List[str]] = None

class CheckoutSuccess(BaseModel):
    status: Literal["checkout"]
    checkout_session_id: str
    url: str
    customer_id: Optional[str] = None
    created_product_id: Optional[str] = None
    created_price_id: Optional[str] = None

class PortalRedirect(BaseModel):
    status: Literal["portal_redirect"]
    reason_code: str
    message: str
    portal_url: str
    subscription_id: str
    configuration_id: str

CheckoutOrPortal = Annotated[Union[CheckoutSuccess, PortalRedirect], Field(discriminator="status")]


# --- [NEW] GET /payments/current_plan ---------------------------------------
@app.get("/payments/current_plan", response_model=CurrentPlanResponse)
async def get_current_plan(
    token: str | None = Query(None, description="Access token dell'utente"),
):
    if REQUIRED_AUTH:
        verify_access_token(token, cognito_sdk)

    client = _mk_plans_client(token)

    sub_id = await _find_current_subscription_id(client)
    if not sub_id:
        raise HTTPException(404, "Nessuna subscription attiva trovata")

    # Prendo lo stato risorse per ottenere plan/variant/pricing_method/periodi
    state: ResourcesState = await _sdk(client.get_subscription_resources, sub_id)

    # opzionale: leggo anche la subscription grezza per lo status
    sub = await _sdk(client.get_subscription, sub_id)
    status = (sub or {}).get("status")

    return CurrentPlanResponse(
        subscription_id=sub_id,
        status=status,
        plan_type=state.plan_type,
        variant=state.variant,
        pricing_method=state.pricing_method,
        active_price_id=state.active_price_id,
        period_start=state.period_start,
        period_end=state.period_end,
    )


# --- [NEW] GET /payments/credits --------------------------------------------
# --- [REWRITE] GET /payments/credits  ---------------------------------------
@app.get("/payments/credits", response_model=Dict[str, Any])
async def get_user_credits(
    token: str | None = Query(None, description="Access token dell'utente"),
    subscription_id: str | None = Query(None, description="Se non fornito, uso la subscription viva"),
):
    if REQUIRED_AUTH:
        verify_access_token(token, cognito_sdk)

    client = _mk_plans_client(token)

    sub_id = subscription_id or await _find_current_subscription_id(client)
    if not sub_id:
        raise HTTPException(404, "Nessuna subscription attiva trovata")

    # Legge lo stato risorse via SDK
    state: ResourcesState = await _sdk(client.get_subscription_resources, sub_id)

    # Normalizza ad un dict (funzione già presente nel tuo codice)
    data = _dataclass_to_dict(state) if hasattr(state, "__dataclass_fields__") else (
        getattr(state, "raw", None) or state
    )

    resources = (data.get("resources") or {})
    provided_list  = resources.get("provided")  or []
    used_list      = resources.get("used")      or []
    remaining_list = resources.get("remaining") or []

    def _sum_credits(items) -> float:
        total = 0.0
        for it in items:
            # supporta sia dict sia eventuali oggetti con attributi
            key = (it.get("key") if isinstance(it, dict) else getattr(it, "key", None)) or ""
            if key.lower() != "credits":
                continue
            qty = (it.get("quantity") if isinstance(it, dict) else getattr(it, "quantity", 0)) or 0
            try:
                total += float(qty)
            except Exception:
                pass
        return total

    provided  = _sum_credits(provided_list)
    used      = _sum_credits(used_list)
    remaining = _sum_credits(remaining_list)

    # formato pulito: solo i totali dei crediti
    def _fmt(x: float):
        try:
            return int(x) if float(x).is_integer() else round(float(x), 4)
        except Exception:
            return x

    return {
        "provided_total":  _fmt(provided),
        "used_total":      _fmt(used),
        "remaining_total": _fmt(remaining),
    }


# --- [UPDATED] POST /payments/checkout --------------------------------------
@app.post("/payments/checkout", response_model=CheckoutOrPortal)  # oppure Dict[str, Any]
async def create_checkout_session_variant(body: CheckoutVariantIn):
    """
    Genera una Checkout Session per VARIANTE di catalogo usando lo SDK.

    Ottimizzazioni L2:
      - riuso della Billing Portal configuration via cache (configuration_id),
        così L1 evita di risolvere/creare la configuration su Stripe;
      - popolamento della price-cache (plan_type, variant) -> created_price_id
        per accelerare futuri deeplink di upgrade/downgrade.

    Se il server risponde 409 single_subscription_portal_redirect,
    ritorna un payload 'portal_redirect' con l'URL del Billing Portal.
    """
    if REQUIRED_AUTH:
        verify_access_token(body.token, cognito_sdk)

    print(body.model_dump_json())

    client = _mk_plans_client(body.token)

    # Blocchi usati sia per la UI del Portal che per il fingerprint della config
    plan_type = body.plan_type
    variants_override = VARIANTS_BUCKET
    features = {
        "payment_method_update": {"enabled": True},
        "subscription_update": {
            "enabled": True,
            "default_allowed_updates": ["price"],
            "proration_behavior": "none",
        },
        "subscription_cancel": {"enabled": True, "mode": "immediately"},
    }
    headline = f"{plan_type} – Manage plan"

    # [NEW] tenta riuso configuration_id dal cache L2
    cfg_key = _cfg_key(plan_type, variants_override, features, headline)
    cached_cfg = _config_cache_get(cfg_key)

    # Costruisci il blocco 'portal' per la richiesta
    if cached_cfg:
        portal_block = {"configuration_id": cached_cfg}
    else:
        portal_block = {
            "plan_type": plan_type,
            "variants_override": variants_override,
            "features_override": features,
            "business_profile_override": {"headline": headline},
        }

    req = DynamicCheckoutRequest(
        success_url=(body.success_url or PLANS_SUCCESS_URL_DEF),
        cancel_url=(body.cancel_url or PLANS_CANCEL_URL_DEF),
        plan_type=body.plan_type,
        variant=body.variant,
        locale=body.locale,
        portal=portal_block,
    )

    try:

        out = await _sdk(client.create_checkout, req)

        print("#*" * 12)
        print(out)
        print("#*" * 12)

        # [NEW] se L1 ha restituito la configuration_id e non era in cache, salvala
        try:
            if not cached_cfg and getattr(out, "configuration_id", None):
                _config_cache_put(cfg_key, out.configuration_id)
        except Exception:
            pass

        # [NEW] popola la price cache per (plan_type, variant) -> created_price_id
        try:
            if body.plan_type and body.variant and getattr(out, "created_price_id", None):
                _price_cache_put(body.plan_type, body.variant, out.created_price_id)
        except Exception:
            pass

        return CheckoutSuccess(
            status="checkout",
            checkout_session_id=out.id,
            url=out.url,
            customer_id=out.customer_id,
            created_product_id=out.created_product_id,
            created_price_id=out.created_price_id,
        )

    except ApiError as e:
        # Gestione specifica del redirect al Billing Portal
        if e.status_code == 409 and isinstance(e.payload, dict):
            detail = (e.payload.get("detail") or {}) if isinstance(e.payload, dict) else {}
            if detail.get("code") == "single_subscription_portal_redirect":
                return PortalRedirect(
                    status="portal_redirect",
                    reason_code=detail.get("code", "single_subscription_portal_redirect"),
                    message=detail.get("message", "Aggiorna il piano esistente dal Billing Portal."),
                    portal_url=detail.get("portal_url"),
                    subscription_id=detail.get("subscription_id"),
                    configuration_id=detail.get("configuration_id"),
                )
        # Altri errori → propaghiamo come HTTPException FastAPI
        raise HTTPException(status_code=getattr(e, "status_code", 500), detail=getattr(e, "payload", str(e)))

@app.post("/payments/portal_session", response_model=Dict[str, Any])
async def create_portal_session(body: PortalSessionIn):
    # 1) Autenticazione
    if REQUIRED_AUTH:
        verify_access_token(body.token, cognito_sdk)

    client = _mk_plans_client(body.token)

    # 2) Stato corrente (usa hint se presente per evitare chiamate superflue)
    sub_id = getattr(body, "current_subscription_id", None) or await _find_current_subscription_id(client)


    '''if not sub_id:
        raise HTTPException(404, "Nessuna subscription attiva trovata")'''

    if not sub_id:
        # ▼ View-only: niente update piano, solo PM update & fatture
        plan_type = getattr(body, "current_plan_type", None) or PLANS_DEFAULT_PLAN_TYPE
        selector = PortalConfigSelector(
            plan_type=plan_type,
            # nessun preset/variants_override
            features_override={
                "payment_method_update": {"enabled": True},
                "invoice_history": {"enabled": True},
                "subscription_update": {"enabled": False},
                "subscription_cancel": {"enabled": False},
            },
            business_profile_override={"headline": f"{plan_type} – Manage billing"},
        )
        req = PortalSessionRequest(
            return_url=(body.return_url or RETURN_URL or PLANS_SUCCESS_URL_DEF),
            portal=selector,
            flow_data=None,
        )
        try:
            sess = await _sdk(client.create_portal_session, req)
            return {"portal_session_id": sess.id, "url": sess.url, "configuration_id": sess.configuration_id}
        except ApiError as e:
            raise HTTPException(status_code=getattr(e, "status_code", 500), detail=getattr(e, "payload", str(e)))


    plan_type = getattr(body, "current_plan_type", None)
    if not plan_type:
        state = await _sdk(client.get_subscription_resources, sub_id)
        plan_type = getattr(state, "plan_type", None) or (state.raw.get("plan_type") if hasattr(state, "raw") else None)
        if not plan_type:
            raise HTTPException(409, "Impossibile dedurre il plan_type dalla subscription corrente")

    # 3) Blocchi deterministici per fingerprint/config cache
    variants_override = VARIANTS_BUCKET
    features = {
        "payment_method_update": {"enabled": True},
        "subscription_update": {"enabled": False},
        "subscription_cancel": {"enabled": True, "mode": "immediately"},
        "invoice_history": {"enabled": True},
    }
    headline = f"{plan_type} – Manage billing"

    # 4) Prova riuso configuration_id da cache L2 (fast-path: zero chiamate Stripe in L1)
    cfg_key = _cfg_key(plan_type, variants_override, features, headline)
    cached_cfg_id = _config_cache_get(cfg_key)

    if cached_cfg_id:
        # Se abbiamo una configuration in cache, usiamo direttamente l'ID
        req = PortalSessionRequest(
            return_url=(body.return_url or RETURN_URL or PLANS_SUCCESS_URL_DEF),
            portal=PortalConfigSelector(configuration_id=cached_cfg_id),
            flow_data=None,
        )
    else:
        # Fallback: lascia che L1 risolva/crei la configuration
        req = PortalSessionRequest(
            return_url=(body.return_url or RETURN_URL or PLANS_SUCCESS_URL_DEF),
            portal=PortalConfigSelector(
                plan_type=plan_type,
                variants_override=variants_override,
                features_override=features,
                business_profile_override={"headline": headline},
            ),
            flow_data=None,
        )

    # 5) Crea la session del Billing Portal tramite L1 e aggiorna la cache se serve
    try:
        sess = await _sdk(client.create_portal_session, req)

        # Se abbiamo usato il fallback e L1 ci ha restituito una configuration, salvala in cache
        if not cached_cfg_id and getattr(sess, "configuration_id", None):
            try:
                _config_cache_put(cfg_key, sess.configuration_id)
            except Exception:
                # la cache non deve mai rompere il flusso
                pass

        return {
            "portal_session_id": sess.id,
            "url": sess.url,
            "configuration_id": sess.configuration_id,
        }
    except ApiError as e:
        raise HTTPException(
            status_code=getattr(e, "status_code", 500),
            detail=getattr(e, "payload", str(e)),
        )

# -----------------------------------------------------------------------------
# --- [FIX/OPTIMIZED] POST /payments/deeplink/update --------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# --- [FIX/OPTIMIZED] POST /payments/deeplink/update --------------------------
# -----------------------------------------------------------------------------
@app.post("/payments/deeplink/update", response_model=Dict[str, Any])
async def create_update_deeplink(body: DeeplinkUpdateIn):
    """
    Crea un deeplink "update" al Billing Portal con:
      - filtro varianti coerente con l'intent (upgrade/downgrade/both/none)
      - features impostate in base all'intent
    Ottimizzazioni L2:
      - Hints opzionali dal client (current_subscription_id/plan_type/variant) per evitare round-trip iniziali.
      - Config cache: se troviamo una configuration_id compatibile, la riusiamo inviandola direttamente a L1.
        -> Fast-path in L1, zero round-trip Stripe per la risoluzione della Portal Configuration.
    """
    if REQUIRED_AUTH:
        verify_access_token(body.token, cognito_sdk)

    client = _mk_plans_client(body.token)

    # [HINT] subscription viva (salta list_subscriptions se il client la conosce)
    sub_id = getattr(body, "current_subscription_id", None) or await _find_current_subscription_id(client)
    if not sub_id:
        raise HTTPException(404, "Nessuna subscription attiva trovata")

    # [HINT] plan_type/variant (salta get_subscription_resources se il client li conosce)
    plan_type = getattr(body, "current_plan_type", None)
    variant   = getattr(body, "current_variant", None)

    if not plan_type or not variant:
        state = await _sdk(client.get_subscription_resources, sub_id)
        if not plan_type:
            plan_type = getattr(state, "plan_type", None) or (state.raw.get("plan_type") if hasattr(state, "raw") else None)
        if variant is None:
            variant = getattr(state, "variant", None) or (state.raw.get("variant") if hasattr(state, "raw") else None)

    if not plan_type:
        raise HTTPException(409, "Impossibile dedurre il plan_type dalla subscription corrente")

    # ▼ Catalogo base → filtro per intent
    base_catalog = body.variants_catalog or VARIANTS_BUCKET
    base_filtered = build_variants_for_intent(
        current_variant=variant,
        intent=body.change_intent,
        catalog=base_catalog,
    )

    # Se il client ha passato una override, la riconfiniamo comunque all'intent
    effective_override = body.variants_override or base_filtered
    effective_override = build_variants_for_intent(
        current_variant=variant,
        intent=body.change_intent,
        catalog=effective_override,
    )

    # Features in base all'intent (upgrade: proration immediata; downgrade: a fine periodo; both: create_prorations)
    feat = features_for_update(body.change_intent)

    # Headline del Portal
    headline = f"{plan_type} – Update plan"

    # [CACHE] prova a riusare una configuration_id già compatibile
    cfg_key = _cfg_key(plan_type, effective_override, feat, headline)
    cached_cfg_id = _config_cache_get(cfg_key)

    # Costruisci il blocco portal:
    #   - se c'è cache → usa direttamente configuration_id (fast-path L1)
    #   - altrimenti → passa selector completo e cacha alla risposta
    if cached_cfg_id:
        portal_block: Dict[str, Any] = {"configuration_id": cached_cfg_id}
    else:
        portal_block = {
            "plan_type": plan_type,
            "variants_override": effective_override,
            "features_override": feat,
            "business_profile_override": {"headline": headline},
        }

    # Richiesta a L1 (schema SDK invariato)
    req = PortalUpdateDeepLinkRequest(
        return_url=(body.return_url or RETURN_URL or PLANS_SUCCESS_URL_DEF),
        subscription_id=sub_id,
        portal=portal_block,  # dict compatibile con lo SDK (selector/override lato backend)
    )

    try:
        dl = await _sdk(client.create_deeplink_update, req)

        # [CACHE] se non avevamo cache e L1 ci restituisce la configuration_id, salviamola
        try:
            if not cached_cfg_id and getattr(dl, "configuration_id", None):
                _config_cache_put(cfg_key, dl.configuration_id)
        except Exception:
            # cache best-effort: non bloccare il flusso anche se fallisce
            pass

        return {"deeplink_id": dl.id, "url": dl.url, "configuration_id": dl.configuration_id}

    except ApiError as e:
        # Propaga come HTTPException FastAPI con payload originale
        raise HTTPException(status_code=getattr(e, "status_code", 500), detail=getattr(e, "payload", str(e)))


@app.post("/payments/deeplink/upgrade", response_model=Dict[str, Any])
async def create_upgrade_deeplink(body: DeeplinkUpgradeIn):
    if REQUIRED_AUTH:
        verify_access_token(body.token, cognito_sdk)

    client = _mk_plans_client(body.token)

    # 1) Subscription viva (usa hint se presenti per evitare round-trip)
    hinted_sub_id = getattr(body, "current_subscription_id", None)
    sub_id = hinted_sub_id or await _find_current_subscription_id(client)
    if not sub_id:
        raise HTTPException(404, "Nessuna subscription attiva trovata")

    # 2) Stato corrente (usa hint se presenti)
    current_plan_type = getattr(body, "current_plan_type", None)
    current_variant   = getattr(body, "current_variant", None)
    if not current_plan_type or not current_variant:
        state = await _sdk(client.get_subscription_resources, sub_id)
        current_plan_type = current_plan_type or getattr(state, "plan_type", None) or (state.raw.get("plan_type") if hasattr(state, "raw") else None)
        current_variant   = current_variant   or getattr(state, "variant",   None) or (state.raw.get("variant")    if hasattr(state, "raw") else None)
    if not current_plan_type:
        raise HTTPException(409, "Impossibile dedurre il plan_type dalla subscription corrente")

    # 3) Catalogo per la Portal Configuration: limitiamo a "current + target" se noto
    base_catalog   = VARIANTS_BUCKET
    target_variant = body.target_variant
    if target_variant and current_variant:
        variants_override = _sorted_variants([v for v in base_catalog if v in {current_variant, target_variant}])
    else:
        # Se non conosco la variant target (es. arrivo da target_price_id), mostro comunque current + bucket
        variants_override = _sorted_variants([current_variant] + base_catalog) if current_variant else base_catalog

    # 4) Rilevazione automatica upgrade vs downgrade (se conosco entrambe le varianti)
    #    - upgrade:   target_value > current_value
    #    - downgrade: target_value < current_value
    #    - altrimenti default a "upgrade"
    is_upgrade = True
    if current_variant and target_variant:
        is_upgrade = _variant_value(target_variant) > _variant_value(current_variant)

    # 5) Features del Portal a corredo del deeplink confermato
    features = {
        "payment_method_update": {"enabled": True},
        "subscription_update": {
            "enabled": True,
            "default_allowed_updates": ["price"],
            "proration_behavior": "always_invoice" if is_upgrade else "none",
        },
        "subscription_cancel": {"enabled": True, "mode": "at_period_end"},
    }

    # 6) Sconti fissati lato server (nessun input utente)
    #    - upgrade   → 1%
    #    - downgrade → 99%
    raw_discounts = [RawDiscountSpec.percent(1.0)] if is_upgrade else [RawDiscountSpec.percent(99.0)]

    # 7) Preparazione selector Portal con FAST-PATH via configuration_id cache
    plan_for_target = (body.target_plan_type or current_plan_type)
    headline = f"{current_plan_type} – {'Confirm upgrade' if is_upgrade else 'Confirm downgrade'}"

    cfg_key = _cfg_key(plan_for_target, variants_override, features, headline)
    cached_cfg_id = _config_cache_get(cfg_key)

    if cached_cfg_id:
        selector = PortalConfigSelector(configuration_id=cached_cfg_id)
    else:
        selector = PortalConfigSelector(
            plan_type=plan_for_target,
            variants_override=variants_override,
            features_override=features,
            business_profile_override={"headline": headline},
        )

    # 8) Risoluzione target_price_id con FAST-PATH via price cache (se non passato)
    target_price_id = body.target_price_id
    if not target_price_id and target_variant:
        cached_price = _price_cache_get(plan_for_target, target_variant)
        if cached_price:
            target_price_id = cached_price

    # 9) Costruisci la richiesta tipizzata verso l'API livello 1
    req = PortalUpgradeDeepLinkRequest(
        return_url=(body.return_url or RETURN_URL or PLANS_SUCCESS_URL_DEF),
        subscription_id=sub_id,
        portal=selector,
        # target by price OR by plan+variant
        target_price_id=target_price_id,
        target_plan_type=plan_for_target,
        target_variant=body.target_variant,
        quantity=1,
        # sconti decisi lato server:
        raw_discounts=raw_discounts,
    )

    # 10) Invoca L1 e applica politiche di cache in base alla risposta
    try:
        dl = await _sdk(client.create_deeplink_upgrade, req)

        # Cache configuration_id se non era in cache
        try:
            if not cached_cfg_id and getattr(dl, "configuration_id", None):
                _config_cache_put(cfg_key, dl.configuration_id)
        except Exception:
            pass

        # Cache price_id se L1 lo ha risolto e non lo avevamo
        try:
            resolved_price = getattr(dl, "target_price_id", None)
            if not target_price_id and resolved_price and target_variant:
                _price_cache_put(plan_for_target, target_variant, resolved_price)
                target_price_id = resolved_price
        except Exception:
            pass

        # Risposta completa (mantiene parità funzionale + arricchisce con price_id se noto)
        resp: Dict[str, Any] = {
            "deeplink_id": dl.id,
            "url": dl.url,
            "configuration_id": dl.configuration_id,
            "subscription_id": sub_id,
            "change_kind": "upgrade" if is_upgrade else "downgrade",
            "applied_discount_percent": 1.0 if is_upgrade else 99.0,
        }
        if target_price_id:
            resp["target_price_id"] = target_price_id
        return resp

    except ApiError as e:
        raise HTTPException(status_code=getattr(e, "status_code", 500), detail=getattr(e, "payload", str(e)))


# -----------------------------------------------------------------------------
# --- [FIX] POST /payments/deeplink/cancel ------------------------------------
# -----------------------------------------------------------------------------
@app.post("/payments/deeplink/cancel", response_model=Dict[str, Any])
async def create_cancel_deeplink(body: DeeplinkCancelIn):
    if REQUIRED_AUTH:
        verify_access_token(body.token, cognito_sdk)

    client = _mk_plans_client(body.token)

    sub_id = await _find_current_subscription_id(client)
    if not sub_id:
        raise HTTPException(404, "Nessuna subscription attiva trovata")

    state = await _sdk(client.get_subscription_resources, sub_id)

    # ricava plan_type/variant per preset di fallback
    plan_type = getattr(state, "plan_type", None) or (getattr(state, "raw", {}) or {}).get("plan_type")
    variant   = getattr(state, "variant",   None) or (getattr(state, "raw", {}) or {}).get("variant")
    if not plan_type:
        raise HTTPException(409, "Impossibile dedurre il plan_type dalla subscription corrente")

    # Requisito server: almeno uno tra variants_override o portal_preset.
    # Usiamo SEMPRE variants_override (derivato dal catalogo) per essere espliciti.
    catalog  = body.variants_catalog or VARIANTS_BUCKET
    variants = build_variants_for_intent(current_variant=variant, intent=ChangeIntent.both, catalog=catalog)

    portal_block: Dict[str, Any] = {
        "plan_type": plan_type,
        "variants_override": variants,  # soddisfa il requisito server
        # "portal_preset": body.portal_preset or _variant_to_portal_preset(variant),  # opzionale
        "features_override": {
            "payment_method_update": {"enabled": True},
            "subscription_cancel": {
                "enabled": True,
                "mode": "immediately" if body.immediate else "at_period_end",
            },
        },
        "business_profile_override": {"headline": f"{plan_type} – Cancel subscription"},
    }

    req = PortalCancelDeepLinkRequest(
        return_url=(body.return_url or RETURN_URL or PLANS_SUCCESS_URL_DEF),
        subscription_id=sub_id,
        portal=portal_block,   # ← contiene variants_override esplicito
        immediate=body.immediate,
    )

    try:
        dl = await _sdk(client.create_deeplink_cancel, req)
        return {"deeplink_id": dl.id, "url": dl.url, "configuration_id": dl.configuration_id}
    except ApiError as e:
        raise HTTPException(status_code=getattr(e, "status_code", 500), detail=getattr(e, "payload", str(e)))


