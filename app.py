import streamlit as st

st.title('Супер классные нейросетки')
st.caption('От Димы, Толубая и Серёжи')
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.page_link('pages/dmche_model.py', label='Модель ДимыЧе')
    st.page_link('pages/dmche_model_info.py', label='more', icon='👀')

with col3:
    st.page_link('pages/kudinov_model.py', label='Модель Серёжи', icon='👾')
    st.page_link('pages/kudinov_model_info.py', label='Инфа по модели', icon='👀')

st.divider()
st.image('images/meme.gif')