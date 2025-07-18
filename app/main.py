import asyncio
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
from app.auth_sdk.sdk import CognitoSDK, AccessTokenRequest
from app.system_messages.system_message_1 import SYSTEM_MESSAGE

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
    collection_name: str,
    loader_config_id: str,
    custom_loaders: Optional[Dict[str, str]] = None,
    custom_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> dict:
    """
    Crea il payload JSON per /document_loaders/configure_loader
    basandosi sull'estensione del file e sulle convenzioni già in uso.
    """
    file_type = file.filename.split(".")[-1].lower()

    # mappa estensione → loader
    loaders = {
        "png": "ImageDescriptionLoader",
        "jpg": "ImageDescriptionLoader",
        "jpeg": "ImageDescriptionLoader",
        "avi": "VideoDescriptionLoader",
        "mp4": "VideoDescriptionLoader",
        "mov": "VideoDescriptionLoader",
        "mkv": "VideoDescriptionLoader",
        "default": "UnstructuredLoader",
    }

    # kwargs specifici per ciascun loader (resize, API-key, ecc.)
    kwargs = {
        # immagini
        "png": {
            "openai_api_key": get_random_openai_api_key(),
            "resize_to": (256, 256),
        },
        "jpg": {
            "openai_api_key": get_random_openai_api_key(),
            "resize_to": (256, 256),
        },
        "jpeg": {
            "openai_api_key": get_random_openai_api_key(),
            "resize_to": (256, 256),
        },
        # video
        "avi": {
            "resize_to": [256, 256],
            "num_frames": 10,
            "openai_api_key": get_random_openai_api_key(),
        },
        "mp4": {
            "resize_to": [256, 256],
            "num_frames": 10,
            "openai_api_key": get_random_openai_api_key(),
        },
        "mov": {
            "resize_to": [256, 256],
            "num_frames": 10,
            "openai_api_key": get_random_openai_api_key(),
        },
        "mkv": {
            "resize_to": [256, 256],
            "num_frames": 10,
            "openai_api_key": get_random_openai_api_key(),
        },
        # fallback
        "default": {
            "strategy": "hi_res",
            "partition_via_api": False,
        },
    }

    custom_loaders = custom_loaders if custom_loaders else {}
    custom_kwargs = custom_kwargs if custom_kwargs else {}

    #loaders.update(custom_loaders)
    #kwargs.update(custom_kwargs)

    # merge profondo con eventuali override
    loaders      = deep_merge(loaders, custom_loaders or {})
    kwargs       = deep_merge(kwargs,  custom_kwargs or {})

    # loader e kwargs selezionati
    chosen_loader = loaders.get(file_type, loaders["default"])
    chosen_kwargs = kwargs.get(file_type, kwargs["default"])

    # payload conforme all’endpoint configure_loader
    return {
        "config_id": loader_config_id,
        "path": f"data_stores/data/{context}",
        "loader_map": {file.filename.replace(" ", "_"): chosen_loader},
        "loader_kwargs_map": {file.filename.replace(" ", "_"): chosen_kwargs},
        "metadata_map": {
            file.filename.replace(" ", "_"): {"source_context": context}
        },
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
        "output_store_map": {
            file.filename.replace(" ", "_"): {"collection_name": collection_name}
        },
        "default_output_store": {"collection_name": collection_name},
    }


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
    files = {"file": (file.filename.replace(" ", "_"), file_content, file.content_type)}
    await _post_or_400(client, f"{NLP_CORE_SERVICE}/data_stores/upload", data=data, files=files)

    # ---------- 1. prepare loader ----------------------------------------------
    loader_id       = f"{ctx}{file.filename.replace(' ', '')}_loader"
    coll_name       = f"{ctx}{file.filename.replace(' ', '')}_collection"
    loader_payload  = _build_loader_config_payload(
        ctx,
        file,
        coll_name,
        loader_id,
        custom_loaders=loaders,
        custom_kwargs=loader_kwargs,
    )

    print("#"*120)
    print(loaders)
    print(loader_kwargs)

    print(json.dumps(loader_payload, indent=4))

    await _post_or_400(client, f"{NLP_CORE_SERVICE}/document_loaders/configure_loader", json=loader_payload)

    # lancia il loader in async e aspetta che finisca
    await _post_or_400(
        client,
        f"{NLP_CORE_SERVICE}/document_loaders/load_documents_async/{loader_id}",
        data={"task_id": loader_task_id},
    )
    await _wait_task_done(client, f"{NLP_CORE_SERVICE}/document_loaders/task_status/{loader_task_id}")

    # ---------- 2. config / load vector store ----------------------------------
    _, vect_id = await _ensure_vector_store(ctx, client)  # idempotente

    await _post_or_400(
        client,
        f"{NLP_CORE_SERVICE}/vector_stores/vector_store/add_documents_from_store_async/{vect_id}",
        params={"document_collection": coll_name, "task_id": vector_task_id},
    )

    await _wait_task_done(
        client,
        f"{NLP_CORE_SERVICE}/vector_stores/vector_store/task_status/{vector_task_id}"
    )


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
        token: Optional[str] = Form(None),
        loaders: Optional[str] = Form(None),
        loader_kwargs: Optional[str] = Form(None),
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
    token: Optional[str] = Form(None),
    loaders: Optional[str] = Form(None),
    loader_kwargs: Optional[str] = Form(None),
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

    file_meta = {"description": description} if description else None
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

    if REQUIRED_AUTH:
        verify_access_token(input_data.token, cognito_sdk)

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

    ###########################################
    input_data.system_message = SYSTEM_MESSAGE#
    ###########################################

    system_message = input_data.system_message

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

    tools = [{"name": "VectorStoreTools", "kwargs": {"store_id": vectorstore_id}} for vectorstore_id in vectorstore_ids]

    tools.append({"name": "MongoDBTools",
                  "kwargs": {
                      "connection_string": "mongodb://localhost:27017",
                      "default_database": f"default_db",
                      "default_collection": "default_collection"
                  }})

    # Configurazione della chain
    chain_config = {
        "chain_type": "agent_with_tools",
        "config_id": f"{id_}_config", #_agent_with_tools_config",
        "chain_id": f"{id_}", #_agent_with_tools",
        "system_message": system_message, # #SYSTEM_MESSAGE,
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



class ExecuteChainRequest(BaseModel):
    chain_id: str = Field(..., example="example_chain", title="Chain ID", description="The unique ID of the chain to execute.")
    query: Dict[str, Any] = Field(..., example={"input": "What is my name?", "chat_history": [["user", "hello, my name is mario!"], ["assistant", "hello, how are you mario?"]]}, title="Query", description="The input query for the chain.")
    inference_kwargs: Dict[str, Any] = Field(..., example={}, description="")


@app.post("/stream_events_chain")
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
        "avi":  ["VideoDescriptionLoader"],
        "mp4":  ["VideoDescriptionLoader", "VideoEventDetectionLoader"],
        "mov":  ["VideoDescriptionLoader"],
        "mkv":  ["VideoDescriptionLoader"],
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
            "strategy": {
                "name": "strategy",
                "type": "string",
                "default": "hi_res",
                "items": ["hi_res", "fast", "auto"],
                "example": "fast"
            },
            "partition_via_api": {
                "name": "partition_via_api",
                "type": "boolean",
                "default": False,
                "example": True
            }
        }
    }










# ─────────────────────────────────────────────────────────────────────────────
# COST‑ESTIMATE ENDPOINT – formula, params, params_conditions
# ─────────────────────────────────────────────────────────────────────────────
HIRES_PRICE_PER_PAGE  = float(os.getenv("HIRES_PRICE_PER_PAGE",  "0.01"))   # USD
FAST_PRICE_PER_PAGE   = float(os.getenv("FAST_PRICE_PER_PAGE",   "0.001"))
IMAGE_FLAT_COST_USD   = float(os.getenv("IMAGE_FLAT_COST_USD",   "0.005"))  # USD / img
VIDEO_PRICE_PER_MIN   = float(os.getenv("VIDEO_PRICE_PER_MIN",   "0.10"))   # USD / min
FALLBACK_KB_PER_PAGE  = 100                                                # ↳ csv / txt …

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
PAGE_DOCS = {".pdf", ".pptx", ".docx", ".tif", ".tiff"}                    # estendibile

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
    raw = (kwargs or {}).get("strategy", "hi_res")

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
                    strategy   = strategy,
                    size_bytes = size_b,
                    tokens_est = round(min(size_b, 500_000) / 4),
                    cost_usd   = round(pages * price_page, 4),
                    formula    = "cost = {pages} * {price_per_page}",
                    params     = base_params | {  # unione dizionari (3.9+)
                        "pages"         : pages,
                        "price_per_page": None,
                    },
                    params_conditions = cond,
                ))
            else:
                pages_est = math.ceil(size_b / (FALLBACK_KB_PER_PAGE * 1024))
                results.append(FileCost(
                    filename   = up.filename,
                    kind       = kind,
                    pages      = pages_est,
                    strategy   = strategy,
                    size_bytes = size_b,
                    cost_usd   = round(pages_est * price_page, 4),
                    formula    = ("cost = ceil({size_bytes} / 102400) * {price_per_page}"),
                    params     = base_params | {
                        "size_bytes"     : size_b,
                        #"pages_est"      : pages_est,
                        #"KB_per_page_rule": FALLBACK_KB_PER_PAGE,
                        "price_per_page" : None,
                    },
                    params_conditions = cond #| {
                        #"pages_est": f"ceil(size_bytes / ({FALLBACK_KB_PER_PAGE} KB))"
                    #},
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
GPT4O_IN_PRICE        = float(os.getenv("GPT4O_IN_PRICE",        "0.01"))   # USD / 1k tok
GPT4O_OUT_PRICE       = float(os.getenv("GPT4O_OUT_PRICE",       "0.03"))
GPT4O_MINI_IN_PRICE   = float(os.getenv("GPT4O_MINI_IN_PRICE",   "0.002"))
GPT4O_MINI_OUT_PRICE  = float(os.getenv("GPT4O_MINI_OUT_PRICE",  "0.006"))
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
