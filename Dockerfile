FROM python:3.10.6

WORKDIR /prod

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY api api

COPY ssl ssl

CMD uvicorn api.fast:app --host 0.0.0.0 --port 443 --ssl-keyfile ssl/privkey.pem --ssl-certfile ssl/fullchain.pem
# $DEL_END
