FROM python:3.10

ADD main.py .

COPY requirements.txt .

RUN pip install opencv-python face-recognition

CMD ["python","./main.py"]

