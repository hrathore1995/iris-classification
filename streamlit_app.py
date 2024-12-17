import numpy as np 
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import streamlit as st 

# Loading scaler pickle file
with open('scaler.pkl','rb') as scaler_file:
    scaler=pickle.load(scaler_file)

# Loading model pickle file
with open('model.pkl','rb') as model_file:
    model = pickle.load(model_file)

# Creating streamlit app
st.title('Classification of Specie of Flowers')
st.write('This is a prediction from RandomForest Classifier')

iris = load_iris()
features = iris.feature_names 

# Input fields
sepal_length = st.slider(f'{features[0]} cm',min_value=0.0,max_value=10.0,value=5.0,step=0.1)
sepal_width = st.slider(f'{features[1]} cm',min_value=0.0,max_value=10.0,value=5.0,step=0.1)
petal_length = st.slider(f'{features[2]} cm',min_value=0.0,max_value=10.0,value=5.0,step=0.1)
petal_width = st.slider(f'{features[3]} cm',min_value=0.0,max_value=10.0,value=5.0,step=0.1)

# prediction and displaying the results
if st.button('Predict'):
    input_features = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    scaled_features = scaler.transform(input_features)

    prediction = model.predict(scaled_features)[0]
    result = iris.target_names[prediction]

    st.write(f'The predicted species is **{str(result).upper()}**')


