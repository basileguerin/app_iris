import streamlit as st
from PIL import Image
import numpy as np
import json
import joblib
from functions import preprocess_img    
import tensorflow as tf
import cv2

class ImagePredictionApp:

    def __init__(self):
        self.load_model()

    def load_model(self):
        self.model_left = tf.keras.models.load_model("./models/model_left")
        self.model_left.trainable = False

        self.model_right = tf.keras.models.load_model("./models/model_right")
        self.model_right.trainable = False

        self.eye_classifier = tf.keras.models.load_model("./models/eye_classifier")
        self.eye_classifier.trainable = False

        self.encoder = joblib.load("./encoder")

    def predict_image(self, image):
        image_prep = preprocess_img(image)
        eye_prediction = np.argmax(self.eye_classifier.predict(np.array([image_prep])))
        if eye_prediction > 0.5:
            selected_model = self.model_right
            eye = 'Droit'
        else:
            selected_model = self.model_left
            eye = 'Gauche'

        employee_prediction = selected_model.predict(np.array([image_prep]))

        with open("./employees_infos.json", 'r') as f:
            employees_info = json.load(f)

        n_employee = self.encoder.inverse_transform([np.argmax(employee_prediction)])[0]
        employee_info = employees_info.get(str(n_employee))

        return f"""Prédiction du modèle :
               \n Nom : {employee_info['nom']}
               \n Année d'embauche : {employee_info['annee_embauche']}
               \n Genre : {employee_info['genre']}
               \n Poste : {employee_info['poste']}
               \n Oeil : {eye}"""
    
    def detect_eyes_camera(self):
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)

        while run:
            _, frame = camera.read()

            if frame is None:
                st.write("Erreur: Impossible de capturer la vidéo.")
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in eyes:

                cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 2)
                eye_image = frame[y:y + h, x:x + w]
                image_prep = preprocess_img(eye_image)
                eye_prediction = np.argmax(self.eye_classifier.predict(np.array([image_prep])))
                if eye_prediction > 0.5:
                    eye = 'Droit'
                else:
                    eye = 'Gauche'
                cv2.putText(frame, eye, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            FRAME_WINDOW.image(frame, use_column_width=True)
        else:
            st.write('Stopped')

app = ImagePredictionApp()

st.set_page_config(page_title="Application d'Authentification",
                   page_icon=':eye:',
                   layout="wide")

def page_1():
    st.title("Application d'Authentification")
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Sélectionner une image", type=["jpg", "png", "bmp"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Image sélectionnée', use_column_width=False)

    with col2:
        if uploaded_file is not None:
            if st.button('Lancer la Prédiction'):
                prediction_text = app.predict_image(np.array(image))
                st.write(prediction_text)

def page_2():
    st.title("Authentification caméra")
    app.detect_eyes_camera()


def main():
    st.sidebar.title("Navigation")
    pages = ["Images Employés", "Caméra"]
    choix_page = st.sidebar.selectbox("Sélectionnez une page :", pages)

    if choix_page == "Images Employés":
        page_1()
    elif choix_page == "Caméra":
        page_2()

if __name__ == '__main__':
    main()
