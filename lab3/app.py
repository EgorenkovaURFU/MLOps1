import pickle
import pandas as pd
import streamlit as st


model = pickle.load(open('models/model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

def predict(sepal_length_cm, sepal_width_cm, 
            petal_length_cm, petal_width_cm):
    
    data = pd.DataFrame([[sepal_length_cm, sepal_width_cm, 
                                               petal_length_cm, petal_width_cm]],
                                               columns=['sepal length (cm)', 'sepal width (cm)', 
                                                        'petal length (cm)', 'petal width (cm)'])
    scaler.transform(data)
    predictions = model.predict(data)

    return predictions


st.title('This application to predict type of irises')
st.image('images.jpg')
st.header('Fill the form to predict iris:')
sepal_length_cm = st.number_input('вsepal length (cm) :', min_value=0, max_value=200, value=1)
sepal_width_cm = st.number_input('sepal width (cm) :', min_value=0, max_value=200, value=1)
petal_length_cm = st.number_input('petal length (cm) :', min_value=0, max_value=200, value=1)
petal_width_cm = st.number_input('petal width (cm) :', min_value=0, max_value=200, value=1)

target_name = ['Setosa', 'Versicolor', 'Virginica']

if st.button('Predict type of irises'):
    p = predict(sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm)
    st.success(f'{target_name[p[0]]}')


