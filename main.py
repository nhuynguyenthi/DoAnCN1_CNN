import streamlit as st # Thư viện để xây dựng giao diện người dùng.
from PIL import Image # thư viện xử lý hình ảnh
#Thư viện để thực hiện web scraping (lấy dữ liệu từ các trang web).
import requests
from bs4 import BeautifulSoup
import tensorflow as tf
import numpy as np

# Load Keras Model
model = tf.keras.models.load_model(r"D:\DA_CN_AI\trained_model.h5")
# Load Labels
with open(r"D:\DoAn_ML\labels.txt") as f:
    labels = f.read().splitlines()

# Labels for fruits and vegetables
fruits = ['banana', 'apple', 'pear', 'grapes', 'orange', 'kiwi', 'watermelon', 'pomegranate', 'pineapple', 'mango']
vegetables = ['cucumber', 'carrot', 'capsicum', 'onion', 'potato', 'lemon', 'tomato', 'raddish', 'beetroot', 'cabbage',
              'lettuce', 'spinach', 'soy bean', 'cauliflower', 'bell pepper', 'chilli pepper', 'turnip', 'corn',
              'sweetcorn', 'sweet potato', 'paprika', 'jalepeno', 'ginger', 'garlic', 'peas', 'eggplant']


# Function to fetch calories from Google
def fetch_calories(prediction):
    try:
        # tạo url tìm kiếm trê gg số calo của thực phẩm dự đoán
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')

        # Check if calories information is available
        calories_div = scrap.find("div", class_="BNeawe iBp4i AP7Wnd")
        if calories_div:
            calories = calories_div.text
            return calories
        else:
            return "Không tìm thấy calo"
    except Exception as e:
        st.error("Xin lỗi: Không tìm thấy calo")
        print(e)


# Function to process image and predict
def process_and_predict_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    predicted_index = np.argmax(predictions) #Tìm chỉ số của lớp có giá trị dự đoán cao nhất
    confidence = predictions[0][predicted_index]  # Lấy độ tin cậy từ giá trị dự đoán
    return labels[predicted_index], confidence


# Streamlit UI
st.title("Nhận dạng phân loại rau củ quả")

# Sidebar
st.sidebar.title("Bảng Dashboard")
app_mode = st.sidebar.selectbox("Chọn Trang", ["Trang chủ", "Giới thiệu dữ liệu", "Dự đoán"])

# Main Page
if app_mode == "Trang chủ":
    st.header("Hệ thống phân loại nhận dạng rau củ quả")
    image_path = r"D:\DA_CN_AI\home_img.jpg"
    st.image(image_path)

# About Project
elif app_mode == "Giới thiệu dữ liệu":
    st.header("Giới thiệu dữ liệu")
    st.subheader("Về dữ liệu Dataset")
    st.text("Tập dữ liệu này chứa hình ảnh của các mặt hàng thực phẩm sau:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code(
        "vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Nội dung")
    st.text("Tập dữ liệu này chứa ba thư mục:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

# Prediction Page
elif app_mode == "Dự đoán":
    st.header("Dự đoán mô hình")
    test_image = st.file_uploader("Chọn 1 ảnh:")

    if test_image is not None:
        st.image(test_image, width=250)
        save_image_path = r'D:\DA_CN_AI\Download_Image' + test_image.name
        with open(save_image_path, "wb") as f:
            f.write(test_image.getbuffer())
        result, confidence = process_and_predict_image(save_image_path)

# khi nhận diện là rau củ thì là bong bóng, quả là bông tuyết

        if result in vegetables:
            st.info('**Loại : vegetables**')
            st.balloons()
        else:
            st.info('**Loại : fruit**')
            st.snow()

        st.success("**Dự đoán : " + result + '**')
        cal = fetch_calories(result)
        if cal:
            st.warning('**' + cal + '**')
        st.subheader("Độ tin cậy dự đoán:")
        st.write(f"Độ tin cậy: {confidence:.2f}")
