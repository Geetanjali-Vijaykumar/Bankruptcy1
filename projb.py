import pandas as pd
import numpy as np
import streamlit as st
import pickle as p

p_out  = open("model_CNB.pkl","rb")
model1 = p.load(p_out)

def user_input_features():
    st.header('User Input Features')
    IR = st.text_input('Industrial Risk', )
    MR=st.text_input('Management Risk',)
    FF=st.text_input('Financial Risk',)
    CR=st.text_input('Credibility',)
    CM=st.text_input('Competitivness',)
    OR=st.text_input('Operating Risk',)
     
    Data = {'IR':IR, 'MR': MR,
            'FF': FF,'CR': CR,
            'CM': CM, 'OR': OR,}
    features = pd.DataFrame(Data,index=[0])
    return features



def main():
    st.title("Bankruptcy")
    #st.sidebar.header('User Input Parameters')

    #IR = st.sidebar.selectbox('Industrial Risk',('1','0.5','0'))
    #MR = st.sidebar.selectbox('Management Risk',('1','0.5','0'))
    #FF = st.sidebar.selectbox('Financial Flexibility',('1','0.5','0'))
    #CR = st.sidebar.selectbox('Credibility',('1','0.5','0'))
    #CM = st.sidebar.selectbox('Competitiveness',('1','0.5','0'))
    #OR = st.sidebar.selectbox('Operating Risk',('1','0.5','0'))
    #result=""

    df=user_input_features()
    st.write(df)
    result = ""
    if st.button("Predict"):
        result = model1.predict(df)
        st.success('The prediction is {}'.format(result))
        #st.success("hello")
        
        #result=model1.predict_bank(IR,MR,FF,CR,CM,OR)
    #st.sucess('The output is {}'.format(result))
    


if __name__=='__main__':
    main()
