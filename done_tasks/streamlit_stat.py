import streamlit as st
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, poisson
import plotly.graph_objects as go
from scipy import stats

st.title('Визуализация кривых распределений для разных типов распределений')
st.sidebar.header('Какой график показать?')

def show_plots(parameters):
    plot = st.sidebar.selectbox('Выберите распределение', ('Нормальное', 'Экспоненциальное', 'Пуассона'))

    if plot == 'Нормальное':
        st.header('Нормальное распределение')
        mu = st.sidebar.slider('Среднее (mu)', 0.0, 100.0, 50.0)
        sigma = st.sidebar.slider('Стандартное отклонение (sigma)', 0.0, 100.0, 50.0)
        parameters = (mu, sigma)

    elif plot == 'Экспоненциальное':
        st.header('Экспоненциальное распределение')
        scale = st.sidebar.slider('Параметр масштаба (scale)', 0.0, 100.0, 50.0)
        parameters = scale

    elif plot == 'Пуассона':
        st.header('Распределение Пуассона')
        mu = st.sidebar.slider('Среднее (mu)', 0.0, 100.0, 10.0)
        parameters = mu

    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000) if plot == 'Нормальное' else np.linspace(0, 100, 1000)
    
    if plot == 'Нормальное':
        pdf = norm.pdf(x, loc=mu, scale=sigma)
        # fig = plt.figure(figsize=(10, 5))
        # plt.plot(x, pdf)
        # plt.title('Нормальное распределение')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='PDF'))
        fig.update_layout(title='Нормальное распределение', xaxis_title='Значение случайной величины', yaxis_title='Плотность вероятности')
        st.plotly_chart(fig)
    elif plot == 'Экспоненциальное':
        pdf = expon.pdf(x, scale=scale)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='PDF'))
        fig.update_layout(title='Экспоненциальное распределение', xaxis_title='Значение случайной величины', yaxis_title='Плотность вероятности')
        st.plotly_chart(fig)
        # fig2 = plt.figure(figsize=(10, 5))
        # plt.plot(x, pdf)
        # plt.title('Экспоненциальное распределение')
    elif plot == 'Пуассона':
        x = np.arange(0, 20)
        pmf = poisson.pmf(x, mu=mu)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=pmf, name='PMF'))
        fig.update_layout(title='Распределение Пуассона', xaxis_title='Значение случайной величины', yaxis_title='Вероятность')
        st.plotly_chart(fig)
        # fig3 = plt.figure(figsize=(10, 5))
        # plt.plot(x, pmf)
        # plt.title('Распределение Пуассона')
        #plt.xlabel('Значение случайной величины')
        #plt.ylabel('Плотность вероятности')

    # st.pyplot()
show_plots(None)
