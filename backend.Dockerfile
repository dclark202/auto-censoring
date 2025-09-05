# backend.Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg

WORKDIR /code

COPY ./requirements-backend.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# --- OPTIMIZED COPY ---
# Copy only the files and folders needed by the backend
COPY ./backend.py /code/backend.py
COPY ./lora_config/ /code/lora_config/

EXPOSE 8000

# Run the backend server
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]