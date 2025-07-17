FROM python:3.13.3-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --upgrade pip

COPY ./src ./src

RUN pip install -e ./src

COPY ./explainers/growing_spheres/. .

RUN pip install -r ./req.txt

EXPOSE 8000

CMD ["python", "main.py"]