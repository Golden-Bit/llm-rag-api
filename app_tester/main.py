from app_sdk.sdk import LLMRAGClient


def main():
    # Inizializza il client puntando all'URL dove gira la tua app
    #client = LLMRAGClient(base_url="http://localhost:8000/llm-rag")
    client = LLMRAGClient(base_url="https://teatek-llm.theia-innovation.com/llm-rag")
    # 1) Crea un contesto
    create_resp = client.create_context(username="mario", token="abc123", context_name="documenti-finanziari", description="Materiale sensibile")
    print("Context created:", create_resp)

    # 2) Lista contesti per utente
    all_contexts = client.list_contexts(username="mario", token="abc123")
    print("List contexts:", all_contexts)

    # 3) Carica un file
    upload_resp = client.upload_file(
        file_path="/percorso/locale/contratto.pdf",
        contexts=["mario-documenti-finanziari"],
        description="Contratto PDF"
    )
    print("Upload result:", upload_resp)

    # 4) Lista file
    files_in_context = client.list_files(contexts=["mario-documenti-finanziari"])
    print("Files in context:", files_in_context)

    # 5) Configura e carica chain
    chain_config_resp = client.configure_and_load_chain(contexts=["mario-documenti-finanziari"], model_name="gpt-4o-mini")
    print("Chain config:", chain_config_resp)

    # 6) Esegui la chain
    chain_id = "mariodocumenti-finanziari_agent_with_tools"  # potrebbe variare a seconda della logica server
    query_data = {
        "input": "Dammi un riassunto del file caricato",
        "chat_history": []
    }
    execution_resp = client.execute_chain(chain_id=chain_id, query_data=query_data)
    print("Chain execution:", execution_resp)

    # 7) Esecuzione in streaming (esempio)
    print("Chain streaming response:")
    for chunk in client.stream_chain(chain_id=chain_id, query_data=query_data):
        print(chunk, end="")  # i chunk contengono porzioni di testo

    # 8) Download file
    client.download_file(file_id="mario-documenti-finanziari/contratto.pdf", destination="contratto_scaricato.pdf")
    print("File scaricato con successo!")

    # 9) Elimina file (opzione A: via file_id)
    # client.delete_file(file_id=upload_resp["file_id"])
    #  o opzione B: via path
    client.delete_file(file_path="mario-documenti-finanziari/contratto.pdf")

    # 10) Elimina contesto
    client.delete_context("mario-documenti-finanziari")

if __name__ == "__main__":
    main()
