FROM python:3.9
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN pip install wandb
COPY ml-ops-project/src/data/training_model.py training_model.py
ENTRYPOINT ["python", "-u", "training_model.py"]