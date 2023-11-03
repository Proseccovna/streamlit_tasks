import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

st.title('Логистическая регрессия в Streamlit')
st.sidebar.header('Загрузите файл .csv для обучающей выборки')
uploaded_train_file = st.sidebar.file_uploader("Выберите файл .csv для обучающей выборки", type=["csv"])

st.sidebar.header('Загрузите файл .csv для тестовой выборки')
uploaded_test_file = st.sidebar.file_uploader("Выберите файл .csv для тестовой выборки", type=["csv"])

if uploaded_train_file is not None and uploaded_test_file is not None:
    train_data = pd.read_csv(uploaded_train_file, index_col=0)
    test_data = pd.read_csv(uploaded_test_file, index_col=0)
    st.write('Первые 5 строк из обучающей выборки:')
    st.write(train_data.head())
    st.write('Первые 5 строк из тестовой выборки:')
    st.write(test_data.head())

    # Выбор признаков и целевой переменной
    st.sidebar.header('Выберите фичи и таргетную переменную')
    feature_cols = st.sidebar.multiselect('Выберите фичи для регрессии:', train_data.columns)
    target_col = st.sidebar.selectbox('Выберите таргетную переменную:', train_data.columns)

    # Регрессия
    if st.sidebar.button('Выполнить логистическую регрессию'):
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Вывод результатов регрессии
        st.subheader('Результаты регрессии:')
        coefficients = dict(zip(feature_cols, model.coef_))
        coefficients['Смещение'] = model.intercept_
        st.write('Коэффициенты регрессии:')
        st.write(coefficients)

        # Метрики классификации
        y_pred = model.predict(X_test)
        y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        st.header('Метрики классификации:')
        st.write(f'Точность: {precision:.2f}')
        st.write(f'Полнота: {recall:.2f}')
        st.write(f'F1: {f1:.2f}')
        st.write(f'Точность: {accuracy:.2f}')
        
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
        scatter_fig = px.scatter(train_data, x=x_axis, y=y_axis)
        st.plotly_chart(scatter_fig)

    elif plot_type == 'Гистограмма':
        st.subheader('Гистограмма')
        x_axis = st.sidebar.selectbox('Выберите ось X:', feature_cols)
        y_axis = st.sidebar.selectbox('Выберите ось Y:', feature_cols)
        bar_fig = px.bar(train_data, x=x_axis, y=y_axis)
        st.plotly_chart(bar_fig)

    elif plot_type == 'Линейный':
        st.subheader('Линейный')
        x_axis = st.sidebar.selectbox('Выберите ось X:', feature_cols)
        y_axis = st.sidebar.selectbox('Выберите ось Y:', feature_cols)
        line_fig = px.line(train_data, x=x_axis, y=y_axis)
        st.plotly_chart(line_fig)