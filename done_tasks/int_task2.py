import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly_express as px
from plotly.subplots import make_subplots

from plotly.tools import mpl_to_plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go


st.title('БД "Tips" в Streamlit')
tips = pd.read_csv('/home/tata/DS_bootcamp/ds-phase-0/learning/tasks/week-02/datasets/tips.csv', index_col = 0)
st.header('Первые 10 строк таблицы "Tips"')
st.write(tips.head(10))
#График №1

st.header('Гистограмма "Размеры счетов"')

fig1 = px.bar(tips, x=tips.index, y='total_bill', color='total_bill',
              labels={'x': 'Номер счета', 'y': 'Сумма, $'}, title='Размеры счетов')
fig1.update_layout(xaxis_tickangle=-90)
st.plotly_chart(fig1)

#График №2

st.header('Точечный график "Размер чаевых в зависимости от размера счета"')

fig2 = px.scatter(tips, x='total_bill', y='tip', color='tip',
              labels={'total_bill': 'Размер счета, $', 'tip': 'Сумма чаевых, $'}, title='Размер чаевых в зависимости от размера счета')
fig2.update_layout(xaxis_tickangle=-90)
st.plotly_chart(fig2)

#График №3

st.header('Точечный график "Размер чаевых в зависимости от размера счета (+ size)"')

fig3 = px.scatter(tips, x='tip', y='total_bill', color='size',
                  labels={'tip': 'Сумма чаевых, $', 'total_bill': 'Размер счета, $', 'size': 'Размер'})
fig3.update_layout(xaxis_tickangle=-90)
st.plotly_chart(fig3)

#График №4

st.header('Гистограмма "Зависимость суммы счета от дня недели"')

fig4 = px.bar(tips, x='day', y=['total_bill', 'total_bill'], barmode='group',
              labels={'day': 'День недели', 'value': 'Сумма счета', 'variable': 'Значение'},
              title='Зависимость суммы счета от дня недели')
st.plotly_chart(fig4)

# График №5

st.header('Точечный график "Размер чаевых в зависимости от пола клиента"')

fig5 = px.scatter(tips, x='tip', y='day', color='sex',
                  labels={'tip': 'Размер чаевых, $', 'day': 'День недели', 'sex': 'Пол'})
st.plotly_chart(fig5)

# График №6

st.header('Гистограмма "Сумма всех счетов за каждый день по времени (ланч/ужин)"')

fig6 = px.box(tips, x='day', y='total_bill', color='time',
              labels={'day': 'День недели', 'total_bill': 'Сумма счета', 'time': 'Время'})
st.plotly_chart(fig6)

# График №7


st.header('Гистограммы "Чаевые на ланч и ужин"')

lunch_data = tips[tips['time'] == 'Lunch']
dinner_data = tips[tips['time'] == 'Dinner']
fig7 = make_subplots(1, 2, subplot_titles=('Ланч', 'Ужин'))


fig7.add_trace(go.Histogram(x=lunch_data['tip'], nbinsx=10, name='Ланч', marker_color='blue'),
               row=1, col=1)

fig7.add_trace(go.Histogram(x=dinner_data['tip'], nbinsx=10, name='Ужин', marker_color='purple'),
               row=1, col=2)

fig7.update_layout(showlegend=False)
fig7.update_xaxes(title_text='Чаевые', row=1, col=1)
fig7.update_xaxes(title_text='Чаевые', row=1, col=2)
fig7.update_yaxes(title_text='Частота', row=1, col=1)
fig7.update_yaxes(title_text='Частота', row=1, col=2)

st.plotly_chart(fig7)


# График №8

st.header('Точечный график "Зависимость размера счета и чаевых, от пола клиента с разбивкой по курящим/некурящим."')

fig8 = px.scatter(tips, x='total_bill', y='tip', color='smoker', facet_col='sex',
                  labels={'total_bill': 'Сумма счета', 'tip': 'Сумма чаевых', 'smoker': 'Курящий',
                          'sex': 'Пол'})
fig8.update_traces(marker=dict(size=8))
st.plotly_chart(fig8)