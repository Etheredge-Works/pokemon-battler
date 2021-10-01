ARG BASE_IMAGE=python:3.8-buster
FROM $BASE_IMAGE

# install project requirements
COPY src/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# Temp workaround till fix is released
COPY poke-env /tmp/poke-env
RUN pip install -e /tmp/poke-env

COPY . /app
WORKDIR /app

CMD ["kedro", "run"]
