FROM python:3.9-slim

WORKDIR /web

COPY . ./

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "Predictor.py"]
