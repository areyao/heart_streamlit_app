FROM python:3.9-slim

ENV PATH /usr/local/bin:$PATH

EXPOSE 8501

WORKDIR /web

COPY . ./
RUN pip3 install -r requirements.txt

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Predictor.py", "--server.port=8501", "--server.address=0.0.0.0"]