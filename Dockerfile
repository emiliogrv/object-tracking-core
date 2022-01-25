FROM python:3.9.9-buster

WORKDIR app

# Installing deps
RUN apt update

# Files mimetypes check
RUN apt install libmagic1 -y

# GStreamer
RUN apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio -y

# FFmeg
RUN apt install ffmpeg -y

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

CMD gunicorn \
  --bind 0.0.0.0:$FLASK_PORT \
  flask_server:app
