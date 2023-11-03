import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title('Логистическая регрессия в Streamlit')
st.sidebar.header('Загрузите файл .csv')
uploaded_file = st.sidebar.file_uploader("Выберите файл .csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, index_col=0)
    st.write('Первые 5 строк из загруженного файла:')
    st.write(data.head())

    # Выбор признаков и целевой переменной
    st.sidebar.header('Выберите фичи и таргетную переменную')
    feature_cols = st.sidebar.multiselect('Выберите фичи для регрессии:', data.columns)
    target_col = st.sidebar.selectbox('Выберите таргетную переменную:', data.columns)

    # Регрессия
    if st.sidebar.button('Выполнить логистическую регрессию'):
        X = data[feature_cols]
        y = data[target_col]
        model = LinearRegression()
        model.fit(X, y)

        # Вывод результатов регрессии
        st.subheader('Результаты регрессии:')
        coefficients = dict(zip(feature_cols, model.coef_))
        coefficients['Смещение'] = model.intercept_
        st.write('Коэффициенты регрессии:')
        st.write(coefficients)

        st.sidebar.header('Создание графиков')
    plot_type = st.sidebar.selectbox('Выберите тип графика:', ['Точечный', 'Гистограмма', 'Линейный'])
    
    # Определение переменных для графиков на более высоком уровне
    scatter_fig = None
    bar_fig = None
    line_fig = None

    if plot_type == 'Точечный':
        st.subheader('Точечный')
        x_axis = st.sidebar.selectbox('Выберите ось X:', feature_cols)
        y_axis = st.sidebar.selectbox('Выберите ось Y:', feature_cols)
        scatter_fig = px.scatter(data, x=x_axis, y=y_axis)
        st.plotly_chart(scatter_fig)

    elif plot_type == 'Гистограмма':
        st.subheader('Гистограмма')
        x_axis = st.sidebar.selectbox('Выберите ось X:', feature_cols)
        y_axis = st.sidebar.selectbox('Выберите ось Y:', feature_cols)
        bar_fig = px.bar(data, x=x_axis, y=y_axis)
        st.plotly_chart(bar_fig)

    elif plot_type == 'Линейный':
        st.subheader('Линейный')
        x_axis = st.sidebar.selectbox('Выберите ось X:', feature_cols)
        y_axis = st.sidebar.selectbox('Выберите ось Y:', feature_cols)
        line_fig = px.line(data, x=x_axis, y=y_axis)
        st.plotly_chart(line_fig)
