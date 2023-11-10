FROM continuumio/miniconda3

WORKDIR /home/app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install libgl1 -y

COPY . .

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "app.py"]

#docker run -it --rm -p 8501:8501 basileguerin/app_iris
