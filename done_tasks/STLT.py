import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

import streamlit as st
import plotly_express as px
from plotly.tools import mpl_to_plotly
from skimage import io



st.title('Изменение картинки в Streamlit')


st.sidebar.header('Давайте изменим вашу картинку')
top_k = st.sidebar.slider('Выберите количество сингулярных чисел', 0, 1000, 500)

uploaded = st.sidebar.file_uploader('Загрузить картинку', type = ['jpg', 'jpeg', 'png'])

if uploaded is not None:
    
    image = cv2.imdecode(np.fromstring(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.xticks([])
    plt.yticks([])
    U, sing_values, V = np.linalg.svd(image)
    sigma = np.zeros(shape=image.shape)
    np.fill_diagonal(sigma, sing_values)
    trunc_U = U[:, :top_k]
    trunc_sigma = sigma[:top_k, :top_k]
    trunc_V = V[:top_k, :]
    image = trunc_U@trunc_sigma@trunc_V
    image = (image - image.min()) / (image.max() - image.min())
    parameters = top_k
else: 
    image = io.imread('https://upload.wikimedia.org/wikipedia/commons/9/9a/%D0%9D%D0%B5%D1%82_%D1%84%D0%BE%D1%82%D0%BE.png')
    plt.xticks([])
    plt.yticks([])

st.image(image, caption='Ваша картинка', use_column_width=True)
