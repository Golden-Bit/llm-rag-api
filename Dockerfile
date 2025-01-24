# Usa l'immagine ufficiale di Python come base
FROM python:3.10-slim

# Imposta il maintainer (facoltativo)
LABEL maintainer="tuo_nome@example.com"

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Copia il file requirements.txt e installa le dipendenze
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il codice sorgente dell'app nella directory /app
COPY . .

# Espone la porta su cui verrà eseguito FastAPI
EXPOSE 8085

# Comando per avviare l'applicazione FastAPI con Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080". "--workers", "1"]
