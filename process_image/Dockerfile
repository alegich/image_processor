#FROM python:3.12-slim
FROM animcogn/face_recognition:cpu

#RUN apt-get update && apt-get install -y libgl1
RUN pip install flask opencv-python-headless

COPY image_processor_api.py /app/
WORKDIR /app

EXPOSE 5005

CMD ["python", "image_processor_api.py"]
