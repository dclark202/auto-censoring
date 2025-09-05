# frontend.Dockerfile
FROM python:3.11-slim

WORKDIR /code

COPY ./requirements-frontend.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app.py /code/app.py
COPY ./static /code/static/
COPY ./templates /code/templates/

EXPOSE 7860

# Run the frontend server with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]