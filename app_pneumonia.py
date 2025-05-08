import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import streamlit as st
from PIL import Image

def main():
    def set_background_image(image_url):
        st.markdown(
            f"""
               <style>
               .stApp {{
                   background-image: url("{image_url}");
                   background-size: cover;
                   background-position: center;
                   background-repeat: no-repeat;
                   background-attachment: fixed;
               }}
               </style>
               """,
            unsafe_allow_html=True)

    set_background_image('https://i.pinimg.com/736x/2b/7a/0c/2b7a0c359c7fcc18a073eda5790332c4.jpg')

    # page title and header image:
    try:
        st.title("PNEUMONIA DETECTION FROM X-RAY IMAGES")


        # img = Image.open("")
        # st.image(img, width=500)
    except FileNotFoundError:
        st.warning(
            "Image file 'image.jpg' not found. Please ensure the file exists in the correct path.")

    st.sidebar.title("Select Page:")
    page = st.sidebar.radio("Go to", ["Home", "About", "Prediction", "Contact"])

    st.markdown("""
                <style>
                div[data-baseweb="select"] {
            background-color: #000033;

            border-radius: 2px;
            padding: 2px;
        }

                 # @keyframes zoomEffect {
                 #        0% { background-size: 150%; }
                 #        100% { background-size: 140%; }
                 #    }
                    [data-testid="stSidebar"] {
                        background: url("https://i.pinimg.com/736x/70/b4/16/70b416c318472c7a288c020e149d493c.jpg") no-repeat center center;
                        background-size: cover;
                        margin-top: 107px;
                        margin-bottom:-50px;
                        #padding-bottom: 300px;
                        animation: zoomEffect 0.45s infinite alternate;
                    }
                   </style>
            """, unsafe_allow_html=True)

    # Home Page
    if page == "Home":
        st.title("Welcome to Pneumonia Detection App 🩺")
        st.markdown("""
        This app uses a deep learning model to analyze chest X-ray images and predict whether a patient has pneumonia. 
        It is designed to assist healthcare professionals in making quick and accurate diagnoses.
        """)



        st.subheader("Key Features:")
        st.markdown("""
        - **Upload and analyze chest X-ray images.**
        - **View prediction results with confidence scores.**
        - **Learn more about pneumonia and the technology behind the app.**
        """)

        st.markdown("""
        ### Get Started
        Click on the **Prediction** page in the sidebar to upload an X-ray image and see the results!
        """)

    elif page == "About":
        st.title("About This App 🧠")
        st.markdown("""
        This app was developed to assist in the early detection of pneumonia using chest X-ray images. 
        Pneumonia is a serious respiratory condition, and early diagnosis can significantly improve patient outcomes.
        """)

        st.subheader("About Pneumonia")
        st.markdown("""
        Pneumonia is an infection that inflames the air sacs in one or both lungs. 
        Common symptoms include cough, fever, and difficulty breathing. Early detection is crucial for effective treatment.
        """)

        st.subheader("Dataset")
        st.markdown("""
        The model was trained on the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from Kaggle. 
        The dataset contains over 5,000 X-ray images, categorized into two classes: **Normal** and **Pneumonia**.
        """)

        st.subheader("Technology Stack")
        st.markdown("""
        - **Streamlit**: For building the web app.
        - **TensorFlow/Keras**: For training the deep learning model.
        - **Python**: For backend development.
        """)

        st.subheader("Model Performance")
        st.markdown("""
        The model achieved an accuracy of **80%** on the test set, with a precision of **91%** and recall of **93%**.
        """)

        st.subheader("Developer")
        st.markdown("""
        This app was developed by **SHAHABAS ALI**. 
        You can find more about my work on [GitHub](https://github.com/shahabas123).
        """)


    elif page == "Prediction":
        st.title("Pneumonia Prediction 🖼️")
        st.markdown("Upload a chest X-ray image to check for pneumonia.")

        model = load_model('pneumonia_model.h5')

        def predict_pneumonia(image):
            image = image.convert("L")
            image = image.resize((150, 150))
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=-1)
            prediction = model.predict(image)[0][0]
            return prediction

        uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)

            st.write("Analyzing the image...")
            prediction = predict_pneumonia(image)

            class_names = ["NORMAL", "PNEUMONIA"]
            predicted_class = "PNEUMONIA" if prediction>0.5 else "NORMAL"
            #confidence = np.max(prediction) * 100

            # Display the result
            st.success(f"Prediction: {predicted_class}")















    elif page == "Contact":
        st.title("Contact Me 📧")
        st.markdown("""
        Have questions or feedback? Feel free to reach out!
        """)
        st.write("Email: shahabasali751@gmail.com.com")
        st.write("GitHub: [GitHub](https://github.com/shahabas123)")
        st.write("LinkedIn: [LinkedIn](https://www.linkedin.com/in/shahabas-ali-8-/)")


main()