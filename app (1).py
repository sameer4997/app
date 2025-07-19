#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.datasets import load_iris
import pandas as pd


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

df.head()


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

X = df[iris.feature_names]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


joblib.dump(model, "iris_model.pkl")
print("Model saved as iris_model.pkl")


# In[13]:


app_code = '''
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

model = joblib.load("iris_model.pkl")
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

st.set_page_config(page_title="Iris Classifier", layout="centered")
st.title("Iris Flower Prediction App")
st.markdown("Predict the species of Iris flower using a trained Random Forest model.")

st.sidebar.header("Input Features")

def get_input():
    sepal_length = st.sidebar.slider("Sepal length (cm)", 4.0, 8.0, 5.4)
    sepal_width = st.sidebar.slider("Sepal width (cm)", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Petal length (cm)", 1.0, 7.0, 1.3)
    petal_width = st.sidebar.slider("Petal width (cm)", 0.1, 2.5, 0.2)
    data = {
        "sepal length (cm)": sepal_length,
        "sepal width (cm)": sepal_width,
        "petal length (cm)": petal_length,
        "petal width (cm)": petal_width,
    }
    return pd.DataFrame(data, index=[0])

input_df = get_input()

st.subheader("Input Data")
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
st.write(f"Predicted Species: **{target_names[prediction[0]]}**")

st.subheader("Prediction Probabilities")
st.write(pd.DataFrame(prediction_proba, columns=target_names))

# Visualization
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Series(iris.target).map(dict(enumerate(target_names)))

st.subheader("Petal Scatterplot")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="petal length (cm)", y="petal width (cm)", hue="species", ax=ax)
plt.title("Petal Length vs Width")
st.pyplot(fig)
'''


# In[15]:


with open("app.py", "w") as f:
    f.write(app_code)

print("Streamlit app code has been saved as 'app.py'")


# In[16]:


import os
print("Files in this directory:", os.listdir())


# In[17]:


# Create requirements.txt for Streamlit Cloud
requirements = '''
streamlit
scikit-learn
pandas
matplotlib
seaborn
joblib
'''

with open("requirements.txt", "w") as f:
    f.write(requirements.strip())

print("requirements.txt created")


# In[18]:


import os
print(os.listdir())


# In[ ]:




