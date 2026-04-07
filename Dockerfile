FROM python:3.11-slim

# Install system dependencies and Ollama
RUN apt-get update && apt-get install -y curl
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pre-download a small open-source model (llama3.2:3b)
RUN ollama serve & sleep 5 && ollama pull llama3.2:3b

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8501
EXPOSE 11434

# Start Ollama in background, then Streamlit
CMD ollama serve & streamlit run app.py --server.port=8501 --server.address=0.0.0.0