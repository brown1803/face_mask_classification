import streamlit as st
from PIL import Image
import cv2
import pickle
import numpy as np

# import các packages cần thiết
from preprocessing import SimplePreprocessor  # Import modul SimplePreprocessor
from datasets import simpledatasetloader  # Import modul simpledatasetloader

# Khởi tạo danh sách nhãn
classLabels = ['with_mask','without_mask']

# Load SVM model
svm_model = pickle.load(open('svm_normal.model', 'rb'))

# Load KNN model
knn_model = pickle.load(open('knn.model', 'rb'))

# Streamlit app
def main():
    st.title("Image Classification")
    st.write("Upload an image and the system will classify it.")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Preprocess the image
        sp = SimplePreprocessor(64, 64)
        preprocessed_image = sp.preprocess(cv_image)

        # Reshape the preprocessed image
        reshaped_image = preprocessed_image.reshape((1, 64*64*3))

        # Tạo bộ nạp dữ liệu ảnh


        # Perform prediction using SVM
        svm_prediction = svm_model.predict(reshaped_image)

        # Perform prediction using KNN
        knn_prediction = knn_model.predict(reshaped_image)

        # Display the predicted labels
        st.write("SVM Prediction:", classLabels[svm_prediction[0]])
        st.write("KNN Prediction:", classLabels[knn_prediction[0]])

if __name__ == '__main__':
    main()
