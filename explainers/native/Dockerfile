FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --upgrade pip

COPY ./src ./src
RUN pip install --user --no-cache-dir -e ./src

COPY ./explainers/native/requirements.minimal.txt ./requirements.txt
RUN pip install --user --no-cache-dir -r ./requirements.txt

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

COPY --from=builder /root/.local /home/app/.local

COPY ./src ./src
COPY ./explainers/native/growing_spheres.py ./explainers/native/growing_spheres.py
COPY ./explainers/native/wachter.py ./explainers/native/wachter.py
COPY ./explainers/native/face.py ./explainers/native/face.py
COPY ./explainers/native/dispatch.sh ./dispatch.sh

USER app

ENV PATH=/home/app/.local/bin:$PATH

EXPOSE 8000

ENTRYPOINT ["./dispatch.sh"]
CMD ["help"]