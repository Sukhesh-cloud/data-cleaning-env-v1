FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r server/requirements.txt

CMD ["python", "-m", "server.app"]