ARG BASE_IMAGE=mcr.microsoft.com/windows/servercore:ltsc2019
FROM $BASE_IMAGE

WORKDIR /home/app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "app.py"]

#docker run -it --rm -p 8501:8501 basileguerin/app_iris
