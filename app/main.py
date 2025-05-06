import asyncio

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Body, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import httpx
import uuid
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import random
import json
from datetime import datetime
from starlette.responses import StreamingResponse

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

def short_hash(obj: dict | str, length: int = 9) -> str:
    """
    Restituisce i primi `length` caratteri dell'hash SHA‑256
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


async def _post_or_400(client, url: str, **kw):
    """POST helper con gestione errore standard."""
    r = await client.post(url, **kw)
    if r.status_code not in (200, 400):
        raise HTTPException(r.status_code, detail=r.text)
    return r


async def _wait_task_done(client, status_url: str, *, poll_secs: float = 2.0):
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


def _build_loader_config_payload(
    context: str,
    file: UploadFile,
    collection_name: str,
    loader_config_id: str,
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
    vector_store_config_id = f"{context}_vector_store_config"
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
    loader_payload  = _build_loader_config_payload(ctx, file, coll_name, loader_id)
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


# --------------------------- REWRITE *ENTIRE* helper ---------------------------
async def upload_file_to_contexts_async(
    file: UploadFile,
    contexts: List[str],
    file_metadata: Optional[Dict[str, Any]] = None,
    *,                       # ← chiamato dall’endpoint /upload
    background_tasks: BackgroundTasks,
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
            client,          # passiamo il client già creato
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
    async with httpx.AsyncClient() as client:
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
                                  file_metadata: Optional[Dict[str, Any]] = None):
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
                    "resize_to": (256, 256)
                },
                "jpg": {
                    "openai_api_key": get_random_openai_api_key(),
                    "resize_to": (256, 256)
                },
                # "txt": {}
                "avi": {
                    "resize_to": [256, 256],  # Dimensione per il resize dei frame
                    "num_frames": 10,  # Numero di frame da estrarre (opzionale)
                    #"frame_rate": 2,           # In alternativa, specificare un frame_rate
                    "openai_api_key": get_random_openai_api_key(),
                },
                "mp4": {
                    "resize_to": [256, 256],  # Dimensione per il resize dei frame
                    "num_frames": 10,  # Numero di frame da estrarre (opzionale)
                    #"frame_rate": 2,           # In alternativa, specificare un frame_rate
                    "openai_api_key": get_random_openai_api_key(),
                },
                "mov": {
                    "resize_to": [256, 256],  # Dimensione per il resize dei frame
                    "num_frames": 10,  # Numero di frame da estrarre (opzionale)
                    #"frame_rate": 2,           # In alternativa, specificare un frame_rate
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
    description: str | None = None

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

    print(request.model_dump_json())

    if REQUIRED_AUTH:
        verify_access_token(request.token, cognito_sdk)

    username = request.username
    token = request.token

    print(f"Creating context: {request.context_name} for user: {username}")

    # Aggiungi username nei metadati del contesto
    metadata = {
        "display_name": request.display_name,
        "description": request.description,
        # "owner": username  # Memorizziamo l'username dell'utente che ha creato il contesto
    }  # if request.description else {"owner": username}

    result = await create_context_on_server(f"{username}-{request.context_name}", metadata)
    print(result)
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
        username: Optional[str] = Form(None),
        token: Optional[str] = Form(None)
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

    file_metadata = {"description": description} if description else None
    result = await upload_file_to_contexts(file, contexts, file_metadata)
    return result


@app.post("/upload_async", response_model=FileUploadResponse)
async def upload_file_to_multiple_contexts_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    contexts: List[str] = Form(...),
    description: Optional[str] = Form(None),
    username: Optional[str] = Form(None),
    token: Optional[str] = Form(None),
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

    file_meta = {"description": description} if description else None
    return await upload_file_to_contexts_async(
        file,
        contexts,
        file_meta,
        background_tasks=background_tasks,
    )


# Helper function to list files by context
async def list_files_in_context(contexts: Optional[List[str]] = None):
    async with httpx.AsyncClient() as client:
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
    - se `file_path` è fornito => aggiorna solo quel percorso
    - se `file_id` (UUID) è fornito => aggiorna tutte le copie di quel file nei vari contesti
    Almeno uno dei due parametri è obbligatorio.
    """
    if REQUIRED_AUTH:
        verify_access_token(request.token, cognito_sdk)

    if not request.file_path and not request.file_id:
        raise HTTPException(
            status_code=400,
            detail="Devi fornire `file_path` oppure `file_id`",
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
            raise HTTPException(status_code=e.response.status_code, detail=f"Errore HTTP: {e.response.text}")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Errore interno: {str(e)}")


# Modello di input per la configurazione e il caricamento della chain
class ConfigureAndLoadChainInput(BaseModel):
    contexts: List[str] = []  # Lista di contesti (vuota di default)
    model_name: Optional[str] = "gpt-4o",  # Nome del modello, default "gpt-4o-mini"
    system_message: Optional[str] = "You are an helpful assistant."
    token: Optional[str] = None

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

    # Estrai i valori dal modello
    contexts = input_data.contexts
    model_name = input_data.model_name

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

    # vector_store_config_id = f"{context}_vector_store_config"
    vectorstore_ids = [f"{context}_vector_store" for context in contexts]

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
            raise HTTPException(status_code=e.response.status_code, detail=f"Errore HTTP: {e.response.text}")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Errore interno: {str(e)}")


# Retrieve info associated with a single context (by ID or name)
@app.get("/context_info/{context_name}", response_model=Dict[str, Any])
async def get_context_info(context_name: str, token: str):

    if REQUIRED_AUTH:
        verify_access_token(token, cognito_sdk)

    result = await create_context_on_server(context_name)
    return result


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
                    "Prefissa con loader: o vector: (es.: loader:uuid,vector:uuid)")
):
    """
    Ritorna lo stato corrente di tutti i task richiesti.

    Esempio:
        /tasks_status?tasks=loader:1234,vector:abcd
    """
    print("#" * 120)
    print(tasks)
    print("---")
    task_items: List[TaskStatusItem] = []
    for raw in tasks[0].split(","):
        print(raw)
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
    print("---")
    print(results)
    print("#" * 120)
    # costruiamo la risposta
    statuses: Dict[str, Any] = {}
    for itm, res in zip(task_items, results):
        if isinstance(res, Exception):
            statuses[itm.task_id] = {"status": "ERROR", "error": str(res)}
        else:
            statuses[itm.task_id] = res

    print({
        "requested": len(task_items),
        "statuses": statuses,
        "timestamp": datetime.utcnow().isoformat()
    })


    return {
        "requested": len(task_items),
        "statuses": statuses,
        "timestamp": datetime.utcnow().isoformat()
    }


