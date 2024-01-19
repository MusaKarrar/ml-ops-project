FROM python:3.10-slim
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install wandb
COPY ml-ops-project/src/data/train_model.py train_model.py

ENTRYPOINT ["python", "-u", "src/train_model.py"]