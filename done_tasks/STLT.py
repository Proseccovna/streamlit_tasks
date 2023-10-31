import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import plotly_express as px
from plotly.tools import mpl_to_plotly
from skimage import io
from sklearn.datasets import load_iris # pip install scikit-learn
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

st.title('Изменение картинки в Streamlit')


st.sidebar.header('Давайте изменим вашу картинку')
top_k = st.sidebar.slider('Выберите количество сингулярных чисел', 0, 1000, 500)

uploaded = st.sidebar.file_uploader('Загрузить картинку', type = ['jpg'])

if uploaded is not None:
    
    image = io.imread(uploaded)
    plt.xticks([])
    plt.yticks([])
    U, sing_values, V = np.linalg.svd(image)
    sigma = np.zeros(shape=image.shape)
    np.fill_diagonal(sigma, sing_values)
    trunc_U = U[:, :top_k]
    trunc_sigma = sigma[:top_k, :top_k]
    trunc_V = V[:top_k, :]
    image = trunc_U@trunc_sigma@trunc_V
    image = (image - image.min()) / (image.max() - trunc_image.min())
    parameters = top_k
else: 
    image = io.imread('https://upload.wikimedia.org/wikipedia/commons/9/9a/%D0%9D%D0%B5%D1%82_%D1%84%D0%BE%D1%82%D0%BE.png')
    plt.xticks([])
    plt.yticks([])

st.image(image, caption='Ваша картинка', use_column_width=True)
