
import pandas as pd
import streamlit as st 
from sklearn.naive_bayes import CategoricalNB as CB
from pickle import dump
from pickle import load

st.title('Model Deployment: Naive Bayes-Bankruptcy detection')
st.sidebar.header('User Input Parameters')
def user_input_features():
    IR = st.sidebar.selectbox('Industrial Risk',('1','0.5','0'))
     MR = st.sidebar.selectbox('Management Risk',('1','0.5','0'))
    FF = st.sidebar.selectbox('Financial Flexibility',('1','0.5','0'))
    CR = st.sidebar.selectbox('Credibility',('1','0.5','0'))
    CM = st.sidebar.selectbox('Competitiveness',('1','0.5','0'))
    OR = st.sidebar.selectbox('Operating Risk',('1','0.5','0'))
    data = {'IR':IR,
            'MR':MR,
            'FF':FF,
            'CR':CR,
            'CM':CM,
            'OR':OR}
    features = pd.DataFrame(data,index = [0])
    return features 
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

# load the model from disk
loaded_model = load(open('proj1.sav', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.write(prediction_proba)

