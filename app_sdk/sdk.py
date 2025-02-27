import requests
from typing import Any, Dict, List, Optional, Generator, Union


class LLMRAGClient:
    """
    SSDK per interagire con l'API LLM-RAG (la stessa descritta dal codice FastAPI).
    Fornisce metodi corrispondenti a ciascun endpoint.
    """

    def __init__(self, base_url: str):
        """
        Inizializza il client con un URL base, ad esempio "http://localhost:8000/llm-rag".

        :param base_url: L'endpoint principale dell'app FastAPI (incluso eventuale root_path).
        """
        self.base_url = base_url.rstrip("/")  # Rimuove eventuale slash finale
        self.session = requests.Session()

    # --------------------------------------------------------------------------
    # 1) CREATE CONTEXT - POST /contexts
    # --------------------------------------------------------------------------
    def create_context(self, username: str, token: str, context_name: str, description: Optional[str] = None) -> Dict[
        str, Any]:
        """
        Crea un nuovo contesto (directory) associato ad un utente.

        :param username: Nome dell'utente proprietario del contesto.
        :param token: (Placeholder) Token di autenticazione dell'utente.
        :param context_name: Nome del contesto che si desidera creare.
        :param description: Descrizione del contesto (opzionale).
        :return: Dizionario con 'path' e 'custom_metadata' relativi al contesto creato.
        :raises requests.HTTPError: Se la chiamata HTTP fallisce.
        """
        url = f"{self.base_url}/contexts"
        payload = {
            "username": username,
            "token": token,
            "context_name": context_name
        }
        if description:
            payload["description"] = description

        resp = self.session.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------------------
    # 2) DELETE CONTEXT - DELETE /contexts/{context_name}
    # --------------------------------------------------------------------------
    def delete_context(self, context_name: str) -> Dict[str, Any]:
        """
        Elimina un contesto esistente in base al suo nome (attenzione:
        nel codice FastAPI, 'context_name' potrebbe dover includere l'username se
        non si è rimosso il prefisso altrove).

        :param context_name: Nome o path del contesto da eliminare.
        :return: Dizionario con informazioni sul risultato.
        :raises requests.HTTPError: Se la chiamata HTTP fallisce.
        """
        url = f"{self.base_url}/contexts/{context_name}"
        resp = self.session.delete(url)
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------------------
    # 3) LIST CONTEXTS - POST /list_contexts
    # --------------------------------------------------------------------------
    def list_contexts(self, username: str, token: str) -> List[Dict[str, Any]]:
        """
        Restituisce l'elenco dei contesti (directory) associati a un utente specifico.

        :param username: Nome utente.
        :param token: (Placeholder) Token di autenticazione.
        :return: Lista di contesti (oggetti con 'path' e 'custom_metadata').
        :raises requests.HTTPError: Se la chiamata HTTP fallisce.
        """
        url = f"{self.base_url}/list_contexts"
        payload = {
            "username": username,
            "token": token
        }
        resp = self.session.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------------------
    # 4) UPLOAD FILE - POST /upload
    # --------------------------------------------------------------------------
    def upload_file(
            self,
            file_path: str,
            contexts: List[str],
            description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Carica un file su uno o più contesti.
        Configura automaticamente i loader e indicizza il documento nel vector store.

        :param file_path: Percorso locale del file da caricare.
        :param contexts: Lista di contesti (stringhe).
        :param description: Descrizione opzionale del file.
        :return: Dizionario con file_id e lista dei contesti su cui è stato caricato.
        :raises requests.HTTPError: Se la chiamata HTTP fallisce.
        """
        url = f"{self.base_url}/upload"

        # Prepara form-data
        files = {
            "file": open(file_path, "rb")
        }
        data = {
            "contexts": ",".join(contexts) if len(contexts) > 1 else contexts,
        }
        if description:
            data["description"] = description

        resp = self.session.post(url, files=files, data=data)
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------------------
    # 5) LIST FILES - GET /files
    # --------------------------------------------------------------------------
    def list_files(self, contexts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Recupera la lista dei file in uno o più contesti. Se contexts è None,
        ritorna la lista di tutti i file disponibili.

        :param contexts: Lista di contesti. Se None, li mostra tutti.
        :return: Lista di file. Ogni elemento è un dict con 'path', 'custom_metadata', ecc.
        :raises requests.HTTPError: Se la chiamata HTTP fallisce.
        """
        url = f"{self.base_url}/files"
        params = {}
        if contexts:
            # Aggiunge param ?contexts= per ogni contesto
            # Esempio: /files?contexts=contesto1&contexts=contesto2
            for ctx in contexts:
                # per passare contesti multipli, requests vuole una tupla (k,v)
                # o un list; usiamo un trucco:
                params.setdefault("contexts", [])
                params["contexts"].append(ctx)

        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------------------
    # 6) DELETE FILE - DELETE /files
    # --------------------------------------------------------------------------
    def delete_file(self, file_id: Optional[str] = None, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Elimina un file specificando o il file_id (UUID) o il path completo.
        Se si usa 'file_id', rimuove il file da tutti i contesti in cui è presente.

        :param file_id: UUID del file (se noto).
        :param file_path: Path completo del file nel contesto (es. "user-contesto/filename.pdf").
        :return: Dizionario con un messaggio di dettaglio.
        :raises ValueError: Se né file_id né file_path sono forniti.
        :raises requests.HTTPError: Se la chiamata HTTP fallisce.
        """
        url = f"{self.base_url}/files"
        if not file_id and not file_path:
            raise ValueError("Devi specificare almeno file_id o file_path")

        params = {}
        if file_id:
            params["file_id"] = file_id
        if file_path:
            params["file_path"] = file_path

        resp = self.session.delete(url, params=params)
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------------------
    # 7) CONFIGURE AND LOAD CHAIN - POST /configure_and_load_chain
    # --------------------------------------------------------------------------
    def configure_and_load_chain(
            self,
            contexts: List[str],
            model_name: Optional[str] = "gpt-4o"
    ) -> Dict[str, Any]:
        """
        Configura e carica una chain in memoria basata sui contesti e sul modello LLM specificati.

        :param contexts: Lista di contesti da usare per costruire vector store, ecc.
        :param model_name: Nome del modello LLM (default: "gpt-4o").
        :return: Dizionario con messaggi di successo e dettagli:
                 'message', 'llm_load_result', 'config_result', 'load_result'.
        :raises requests.HTTPError: Se la chiamata HTTP fallisce.
        """
        url = f"{self.base_url}/configure_and_load_chain/"
        payload = {
            "contexts": contexts,
            "model_name": model_name
        }
        resp = self.session.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------------------
    # 8) GET CONTEXT INFO - GET /context_info/{context_name}
    # --------------------------------------------------------------------------
    def get_context_info(self, context_name: str) -> Dict[str, Any]:
        """
        Recupera informazioni su un contesto esistente.

        NOTA: nel codice FastAPI, l'endpoint invoca a sorpresa create_context_on_server(context_name),
              quindi potrebbe risultare in un "falso" create.
              In base al codice presente, potrebbe generare un contesto se non esiste già.
              Usare con cautela o modificare l'implementazione di quell'endpoint sul server.

        :param context_name: Nome (o path) del contesto.
        :return: Dizionario con eventuali metadati sul contesto.
        :raises requests.HTTPError: Se la chiamata HTTP fallisce.
        """
        url = f"{self.base_url}/context_info/{context_name}"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------------------
    # 9) EXECUTE CHAIN - POST /execute_chain
    # --------------------------------------------------------------------------
    def execute_chain(self, chain_id: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Esegue una catena precedentemente caricata, fornendo query e chat_history.
        Risponde in modo sincrono (tutto in una volta).

        :param chain_id: ID della chain da eseguire (es. "mycontext_agent_with_tools").
        :param query_data: Dizionario con l'input utente, es. {"input": "...", "chat_history": [...]}
        :return: Risposta della catena (formato libero, dipende dal server).
        :raises requests.HTTPError: Se la chiamata HTTP fallisce.
        """
        url = f"{self.base_url}/execute_chain"
        params = {"chain_id": chain_id}
        resp = self.session.post(url, params=params, json=query_data)
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------------------
    # 10) STREAM CHAIN - POST /stream_chain
    # --------------------------------------------------------------------------
    def stream_chain(self, chain_id: str, query_data: Dict[str, Any]) -> Generator[str, None, None]:
        """
        Esegue una catena in streaming: ritorna una risposta token-by-token (o chunk-by-chunk).

        :param chain_id: ID della chain.
        :param query_data: Dizionario, tipicamente {"input": "...", "chat_history": [...]}
        :yield: Stringhe (chunk) restituite dal server.
        :raises requests.HTTPError: Se la chiamata HTTP fallisce.
        """
        url = f"{self.base_url}/stream_chain"
        params = {"chain_id": chain_id}
        with self.session.post(url, params=params, json=query_data, stream=True) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk

    # --------------------------------------------------------------------------
    # 11) DOWNLOAD FILE - GET /download
    # --------------------------------------------------------------------------
    def download_file(self, file_id: str, destination: str) -> None:
        """
        Scarica un file (via streaming) dall'endpoint /download e lo salva localmente.

        :param file_id: L'ID o path del file da scaricare.
        :param destination: Percorso locale dove salvare il file.
        :raises requests.HTTPError: Se la chiamata HTTP fallisce.
        """
        url = f"{self.base_url}/download"
        params = {"file_id": file_id}
        with self.session.get(url, params=params, stream=True) as resp:
            resp.raise_for_status()
            with open(destination, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
