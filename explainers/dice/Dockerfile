FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --upgrade pip

COPY ./src ./src
RUN pip install --user --no-cache-dir -e ./src

COPY ./explainers/dice/requirements.minimal.txt ./requirements.txt
RUN pip install --user --no-cache-dir -r ./requirements.txt

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

COPY --from=builder /root/.local /home/app/.local

COPY ./src ./src
COPY ./explainers/dice/dice.py ./explainers/dice/dice.py

USER app

ENV PATH=/home/app/.local/bin:$PATH

EXPOSE 8000

CMD ["celery", "-A", "explainers.dice.dice", "worker", "-l", "info", "-n", "dice_worker@%h", "-Q", "dice,celery"]