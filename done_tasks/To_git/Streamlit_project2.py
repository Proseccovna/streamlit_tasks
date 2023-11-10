import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import streamlit as st
import plotly_express as px
from plotly.tools import mpl_to_plotly
import io
import imageio

def load_data():
    return pd.read_csv('boston.csv')
# Загрузка данных
df = load_data() 

st.title('Работа с датасетом __')

st.sidebar.header('Выберите страницу')
page = st.sidebar.selectbox("Выберите страницу", ["Главная", "Preprocessing и Feature engineering", "Модель обучения", "Итоги"])

# Основная часть приложения в зависимости от выбранной страницы
if page == "Главная":
    st.header('Выполнила команда "LightGBM":')
    st.subheader('🐱Алексей2 (но для нас он №1)')
    st.subheader('🐱Алиса')
    st.subheader('🐱Тата')
    st.header('Исходный датасет')
#     df = pd.read_csv('boston.csv')
    nan_columns = df.columns[df.isna().any()]
    nan_counts = df[nan_columns].isna().sum()
    nan_info_df = pd.DataFrame({'Column': nan_columns, 'NaN Count': nan_counts})
    st.write(nan_info_df)
    
    st.subheader('Размер: 1460х81')
    st.subheader('Количество колонк, содержащих категориальные признаки: 43')
    for i in range(0, len(df.columns), 8):
        interval_columns = df.columns[i:i + 8]
        interval_dtypes = df[interval_columns].dtypes
        interval_dtypes_filtered = interval_dtypes[interval_dtypes.isin(['int64', 'float64'])]
        st.write(pd.DataFrame(interval_dtypes).transpose())


elif page == "Preprocessing и Feature engineering":
    parts = st.sidebar.selectbox("Выберите подраздел", ["Preprocessing", "Feature engineering"])
    if parts == 'Preprocessing':
        st.subheader("Часть 1 - Preprocessing от Алисы")
        st.markdown("*1. Чистка данных от NaN-значений*")
        st.markdown("*2. Борьба против выбросов: все значения, превышающие 99-й перцентиль заменили на его значение*")
        # Очищаем данные от NaN значений для корректного расчета перцентилей
        # df = pd.read_csv('boston.csv')
        df_clean = df['LotFrontage'].dropna()

# Рассчитываем 95-й и 99-й перцентили для 'LotFrontage'
        percentile_95 = np.percentile(df_clean, 95)
        percentile_99 = np.percentile(df_clean, 99)

# Строим диаграмму рассеивания
        fig, ax = plt.subplots(figsize=(10, 5))


        ax.axhline(y=percentile_95, color='red', linestyle='--', label=f'95th percentile: {percentile_95}')
        ax.axhline(y=percentile_99, color='blue', linestyle='--', label=f'99th percentile: {percentile_99}')

# Рассчитываем индексы для x-оси
        index = np.arange(len(df_clean))
        ax.scatter(index, df_clean, alpha=0.5)
        ax.set_title('Lot Frontage Percentiles')
        ax.set_xlabel('Index')
        ax.set_ylabel('Lot Frontage')
        ax.legend()

# Отображаем график
        st.pyplot(fig)
        image1 = imageio.imread('Screenshot1.png')[:, :, :]
        st.image(image1, caption="Caption")
        st.markdown("*3. Кодировка котегориальных признаков и еще кое-что*")


        st.subheader("Часть 2 - Preprocessing от Алексея (aka Повелитель подвалов)")
        st.markdown("*1. Чистка данных от NaN-значений*")
        st.markdown("*2. Кодировка котегориальных признаков и еще кое-что:*")
        st.write("Подвальные дела")
        image4 = imageio.imread('Screenshot4.png')[:, :, :]
        st.image(image4, caption="Caption")

        st.markdown("*3. На случай, если вам нужно достать ванны из подвала:*")
        image5 = imageio.imread('Screenshot5.png')[:, :, :]
        st.image(image5, caption="Caption")


        st.subheader("Часть 3 - Preprocessing от Таты")
        st.markdown("*1. Чистка данных от NaN-значений*")
        st.markdown("*2. Кодировка отегориальных признаков и еще кое-что:*")
        st.write("В  GarageYrBlt сделали разделение на временные отрезки постройки:")
        st.write("1 - ранее 1950")
        st.write("2 - 1951-1970")
        st.write("3 - 1971 -1990")
        st.write("4 - 1991 -2010")
        garage_year = df['GarageYrBlt'] 
        st.subheader("Было")
        fig1 = px.scatter(df, x=df.index, y='GarageYrBlt', color_discrete_sequence=['red'], opacity=0.5)
        fig1.update_layout(xaxis_title='ID', yaxis_title='Year', title='GarageYrBlt')
        st.plotly_chart(fig1)

        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
        bins = [0, 1800, 1950, 1970, 1990, 2010]
        labels = ['0', '1', '2', '3', '4']
        df['GarageYrBlt'] = pd.cut(df['GarageYrBlt'], bins=bins, labels=labels, include_lowest=True)
        garage_year = df['GarageYrBlt'] 
        st.subheader("Стало")
        fig2 = px.scatter(df, x=df.index, y='GarageYrBlt', color_discrete_sequence=['green'])
        fig2.update_layout(xaxis_title='ID', yaxis_title='Year', title='GarageYrBlt')
        st.plotly_chart(fig2)

        st.markdown("*Предложение 2 в 1:*")
        st.write('Схлопнули фичи "для богатых"')
        image2 = imageio.imread('Screenshot2.png')[:, :, :]
        st.image(image2, caption="Caption")
        image9 = imageio.imread('cat3.png')[:, :, :]
        st.image(image9, caption="Caption")
        st.markdown("*Предложение 5 в 1: (спойлер - такое никому не надо)*")
        image3 = imageio.imread('Screenshot3.png')[:, :, :]
        st.image(image3, caption="Caption")
        st.write('Коммент by Алиса четко отражает отношение к этим фичам')
        st.markdown("# **🚀 Вжух!**")
    if parts == 'Feature engineering':
        st.subheader("Добавленные фичи")
        image6 = imageio.imread('Screenshot6.png')[:, :, :]
        st.image(image6, caption="Caption")
        st.markdown("# **🚀 Вжух!**")
elif page == "Модель обучения":
    st.subheader("Леша и Алиса, страдающие над моделью")
    image7 = imageio.imread('crying_cats.png')[:, :, :]
    st.image(image7, caption="Caption")
    st.subheader("Тем временем Тата , пилящая Стримлит")
    image8 = imageio.imread('cat1.png')[:, :, :]
    st.image(image8, caption="Caption")
    
    st.header('Тест моделей:')
    # 1) Запустили Catboost регрессию с параметрами по умолчанию
    st.subheader("1. CatBoost регрессия (по умолчанию)")
    st.text("Результат: 0.1356")

    # 2) Потюнили Catboost через гридсёрч, запустили - 0.1356
    st.subheader("2. Тюнинг CatBoost через GridSearchCV")
    st.markdown("Результат: 0.1356")

    # 3) Поигрались с RF, LinReg, Catboost и другими моделями
    st.subheader("3. Эксперименты с разными моделями")
    st.markdown("Результат: 0.14256-14357")

    # 4) Затюнили через гридсёрч RF
    st.subheader("4. Тюнинг случайного леса (RandomForest) через GridSearchCV")
    st.markdown("Результат: 0.14632")

    # 5) Запустили стекинг (ЛинРег + Кэтбуст + RF) = 0.1327
    st.subheader("5. Запуск Stacking (LinReg + CatBoost + RandomForest)")
    st.markdown("Результат: 0.1327")

    # Заключение
    st.subheader("Заключение")
    st.markdown("Решили дальше поработать с фичами и, пока только, обосрались.")
    st.markdown("# **🚀 Вжух!**")

elif page == "Итоги":
    st.header("Лучшая модель:")
    st.subheader('🐱 *CatBoostRegressor (CatBoost)*)')
    st.header('Параметры модели:')
    st.subheader('*depth=5*') 
    st.subheader('*iterations=1000*')
    st.subheader('*l2_leaf_reg=2*')
    st.subheader('*learning_rate=0.05*')
    st.subheader('*verbose=0*')
    st.header("Лучший результат на Kaggle:")
    image10 = imageio.imread('10.png')[:, :, :]
    st.image(image10, caption="Caption")






