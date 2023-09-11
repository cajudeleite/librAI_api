FROM python:3.10.6

WORKDIR /prod

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY api api

CMD uvicorn api.fast:app --host 0.0.0.0 --port 8000
# $DEL_END
