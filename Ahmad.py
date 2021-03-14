import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from pickle import load
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html


### Engine ########
dataset=pd.read_csv('New Pred.csv')
X_pred=dataset['Price'].values.T
scaler = load(open('scaler.pkl', 'rb'))
x_pred=scaler.transform(X_pred.reshape(1,59))
model=load_model('My Model.h5')
preds=model.predict(x_pred)

a=pd.DataFrame(scaler.inverse_transform(x_pred.T.reshape(1,59)).T).reset_index(inplace=True)
p=pd.DataFrame(preds.T )
eva=pd.concat([a, p], axis=0).reset_index(drop=True)
eva.columns=['Price']
eva= eva[(eva.T != 0).any()].reset_index(drop=True)
inp=eva.index
## Dash

fig = px.line(eva, x=eva.index, y='Price')
fig.show()

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=True) 