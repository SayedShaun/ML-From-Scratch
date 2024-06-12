import streamlit as st
import numpy as np
import joblib

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Email Spam Classifier")
input_email = st.text_area("Leave Your Email Here", height=100)

def user_input(email:str)->np.array:
  vector = joblib.load("Weights/email_spam_tfidf-vectorizer.pkl")
  return vector.transform([email])

def predict(email_vector:np.array)->str:
  model = joblib.load("Weights/email_spam_model.pkl")
  return model.predict(email_vector)[0]


output = predict(user_input(input_email))
check_email = st.button("Check Email", use_container_width=True)

if check_email:
  if output == 1:
    st.write("This is a Spam Email, Be Careful!")
  else:
    st.write("This is a Ham Email, No Problem!")
    
  