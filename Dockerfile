FROM python:3.14-slim
WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements-serving.txt
COPY pipeline ./pipeline
COPY models ./models
COPY serving ./serving
COPY configs ./configs
COPY artifacts/weights.pt artifacts/bundle.pkl artifacts/meta.json ./artifacts/
RUN useradd --create-home --uid 1000 app
USER app
ENV URE_ARTIFACTS=/app/artifacts
EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
