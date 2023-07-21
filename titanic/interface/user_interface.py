import streamlit as st

st.markdown("""# Titanic
## A **titanic** mistake ?
Imagine this is 1912.
You just bought a ticket to board the biggest cruise ship ever made.
Although you aren't a fan of being on the sea or in the water, you were told this ship was unsinkable.
But is it ?
Enter your information below to find out whether you are Jack or Rose.""")

st.number_input('Ticket class:')
st.text_input('Full name:')
st.selectbox('Sex:', ['male', 'female'])
st.number_input('Age:')
st.number_input('Number of siblings or spouses onboard:')
st.number_input('Number of parents or children onboard:')
st.text_input('Ticket number:')
st.number_input('Ticket fare:')
st.text_input('Cabin number:')
st.selectbox('Port of embarkation (S, C or Q):', ['S', 'C', 'Q'])
