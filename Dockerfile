# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501

WORKDIR ./app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/malikberrada/Tribute-To-Cole-Porter-App.git .

RUN apt-get -y update && apt-get -y install freeglut3-dev && apt-get -y install libgtk2.0-dev && apt-get -y install libasound2-dev && apt-get -y install libsndfile1-dev && apt-get -y install ffmpeg

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=107.161.68.105"]