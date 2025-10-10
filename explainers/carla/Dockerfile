FROM python:3.7-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

COPY ./src ./src
COPY ./explainers/carla/requirements.minimal.txt ./requirements.txt

RUN chown -R app:app /app

USER app

ENV PATH=/home/app/.local/bin:$PATH

RUN pip install --no-cache-dir --user -e ./src
RUN pip install --no-cache-dir --user --upgrade --no-deps -r ./requirements.txt

COPY ./explainers/carla/actionable_recourse.py ./explainers/carla/actionable_recourse.py
COPY ./explainers/carla/dice.py ./explainers/carla/dice.py
COPY ./explainers/carla/dispatch.sh ./dispatch.sh

EXPOSE 8000

ENTRYPOINT ["./dispatch.sh"]
CMD ["help"]
