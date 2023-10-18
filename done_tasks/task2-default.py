import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly_express as px
from plotly.tools import mpl_to_plotly

st.title('БД "Tips" в Streamlit')
tips = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
st.header('Первые 10 строк таблицы "Tips"')
st.write(tips.head(10))
#График №1

st.header('Гистограмма "Размеры счетов"')

fig1 = plt.figure(figsize=(10, 5))
plt.bar(tips.index, tips['total_bill'], color = 'green', alpha = 0.7)
plt.xticks(rotation=90)
plt.xlabel('Номер счета')
plt.ylabel('Сумма, $')
plt.title('Размеры счетов')
plt.tight_layout()
st.pyplot(fig1)

#График №2

st.header('Точечный график "Размер чаевых в зависимости от размера счета"')

fig2 = plt.figure(figsize=(10, 5))
plt.scatter(tips['total_bill'], tips['tip'], color = 'blue', alpha = 0.5)
plt.xticks(rotation=90)
plt.xlabel('Размер счета, $')
plt.ylabel('Сумма чаевых, $')
plt.title('Размер чаевых в зависимости от размера счета')
plt.tight_layout()
st.pyplot(fig2)

#График №3

st.header('Точечный график "Размер чаевых в зависимости от размера счета (+ size)"')

fig3 = plt.figure(figsize=(10, 5))
# plt.scatter(tips['total_bill'], tips['tip'], s = tips['size'], color = 'red')
sns.scatterplot(data=tips, x='tip', y='total_bill', hue='size')
plt.xticks(rotation=90)
plt.xlabel('Размер счета, $')
plt.ylabel('Сумма чаевых, $')
plt.title('Размер чаевых в зависимости от размера счета')
plt.tight_layout()
st.pyplot(fig3)

#График №4

st.header('Гистограмма "Зависимость суммы счета от дня недели"')

fig4 = plt.figure(figsize=(10, 5))
sns.barplot(x=tips['day'].unique(), y=tips.groupby('day')['total_bill'].mean(), color = 'violet')
sns.barplot(x=tips['day'].unique(), y=tips.groupby('day')['total_bill'].median(), color = 'pink')
plt.title('Зависимость суммы счета от дня недели')
plt.xlabel('День недели')
plt.ylabel('Сумма счета')
plt.legend(['Среднее', 'Медиана']);
plt.tight_layout()
st.pyplot(fig4)

#График №5

st.header('Точечный график "Размер чаевых в зависимости от пола клиента"')

fig5 = plt.figure(figsize=(10, 5))
sns.scatterplot(data=tips, x='tip', y='day', hue='sex')
plt.title('Размер чаевых в зависимости от пола клиента')
plt.xlabel('Размер чаевых')
plt.ylabel('День недели')
plt.tight_layout()
st.pyplot(fig5)

#График №6

st.header('Гистограмма "Сумма всех счетов за каждый день по времени (ланч/ужин)"')

fig6 = plt.figure(figsize=(10, 5))
plt.title('Сумма всех счетов за каждый день по времени (ланч/ужин)')
plt.xlabel('День недели')
plt.ylabel('Сумма счета')
sns.boxplot(data=tips, x=tips['day'], y=tips['total_bill'], hue='time', color = 'yellow')
plt.tight_layout()
st.pyplot(fig6)

#График №7

st.header('Гистограммы "Чаевые на ланч и ужин"')

lunch_data = tips[tips['time'] == 'Lunch']
dinner_data = tips[tips['time'] == 'Dinner']
fig7, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(lunch_data['tip'], bins=10, kde=True, ax=axes[0], color = 'blue')
axes[0].set_xlabel('Чаевые')
axes[0].set_ylabel('Частота')
axes[0].set_title('Гистограмма чаевых на ланч')
sns.histplot(dinner_data['tip'], bins=10, kde=True, ax=axes[1], color = 'purple')
axes[1].set_xlabel('Чаевые')
axes[1].set_ylabel('Частота')
axes[1].set_title('Гистограмма чаевых на ужин')
plt.tight_layout()
# st.header('Гистограммы "Чаевые на обед и ланч"')
# fig7 = plt.figure(figsize=(10, 5))
# sns.displot(data=tips, x=tips['tip'], kind='hist', col='time')
# plt.xlabel('Сумма чаевых')
# plt.ylabel('Время')
# plt.tight_layout()
st.pyplot(fig7)

#График №8

st.header('Точечный график "Зависимость размера счета и чаевых, от пола клиента с разбивкой по курящим/некурящим."')

fig8 = plt.figure(figsize=(10, 5))
sns.scatterplot(data=tips[tips['sex'] == 'Male'], x=tips['total_bill'], y=tips['tip'], hue='smoker')
sns.scatterplot(data=tips[tips['sex'] == 'Female'], x=tips['total_bill'], y=tips['tip'], hue='smoker')
plt.xlabel('Сумма счета')
plt.ylabel('Сумма чаевых')
plt.tight_layout()
st.pyplot(fig8)

#Создание графика для мужчин

fig_8 = plt.figure(figsize=(10, 5))
sns.scatterplot(data=tips[tips['sex'] == 'Male'], x='total_bill', y='tip', hue='smoker')
plt.xlabel('Сумма счета')
plt.ylabel('Сумма чаевых')
plt.title('Мужчины')
plt.tight_layout()

# Создание графика для женщин

fig_9 = plt.figure(figsize=(10, 5))
sns.scatterplot(data=tips[tips['sex'] == 'Female'], x='total_bill', y='tip', hue='smoker')
plt.xlabel('Сумма счета')
plt.ylabel('Сумма чаевых')
plt.title('Женщины')
plt.tight_layout()

# Отображение графиков в Streamlit
st.subheader('Мужчины')
st.pyplot(fig_8)
st.subheader('Женщины')
st.pyplot(fig_9)

