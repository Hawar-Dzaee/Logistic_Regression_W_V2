import torch 
import torch.nn as nn 
import streamlit as st
import numpy as np 
import plotly.graph_objects as go 


#-------------
class LogisticRegression:

  def __init__(self ,lower_0, upper_0, sample_size_0, noise_0,
                lower_1, upper_1, sample_size_1, noise_1):
        
        self.lower_0 = lower_0
        self.upper_0 = upper_0
        self.sample_size_0 = sample_size_0
        self.noise_0 = noise_0
        self.lower_1 = lower_1
        self.upper_1 = upper_1
        self.sample_size_1 = sample_size_1
        self.noise_1 = noise_1

        self.x0 = torch.linspace(lower_0, upper_0, sample_size_0) + torch.tensor([noise_0])
        self.x1 = torch.linspace(lower_1, upper_1, sample_size_1) - torch.tensor([noise_1])
        self.X = torch.cat((self.x0, self.x1), dim=0)
        self.y = torch.cat((torch.zeros(len(self.x0)), torch.ones(len(self.x1))), dim=0)


  def model(self,w):
        self.w = w 

  def generate_plot(self):
        scatter_class_0 = go.Scatter(
            x=self.x0,
            y=torch.zeros(len(self.x0)),
            mode='markers',
            marker=dict(color='purple'),
            name='class 0'
        )
        scatter_class_1 = go.Scatter(
            x=self.x1,
            y=torch.ones(len(self.x1)),
            mode='markers',
            marker=dict(color='orange'),
            name='class 1'
        )
        
        non_linear_line = go.Scatter(
            x=torch.linspace(-3, 3, 1000),
            y=torch.sigmoid(self.w * torch.linspace(-3, 3, 1000)),
            mode='lines',
            line={'color': 'rgb(27,158,119)'},
            name='model'
        )
        layout = go.Layout(
            xaxis=dict(
                range=[-3.1, 3.1],
                title='X',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='rgba(205, 200, 193, 0.7)'
            ),
            yaxis=dict(
                range=[-0.5, 1.5],
                title='Y',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='rgba(205, 200, 193, 0.7)'
            ),
            height=500,
            width=2600
        )
        figure = go.Figure(data=[scatter_class_0, scatter_class_1, non_linear_line], layout=layout)
        return figure



#---------------------------------------
# streamlit 
  

st.set_page_config(layout='wide')
st.title("Logistic Regression : Weight & Bias")
st.write('By : Hawar Dzaee')


with st.sidebar:

    st.subheader("Data Generation")
    sample_size_0_val = st.slider("sample size Class 0:", min_value= 2, max_value=12, step=1, value= 3)
    sample_size_1_val = st.slider("sample size Class 1:", min_value= 2, max_value=12, step=1, value= 3)


    st.subheader("Adjust the parameters to minimize the loss")
    w_val = st.slider("weight (w):", min_value=-4.0, max_value=18.0, step=0.1, value= -3.5)


container = st.container()


with container:
 

    col1, col2 = st.columns([3,3])

    with col1:
        data = LogisticRegression(lower_0 = -3,upper_0 = -1.5, sample_size_0 = sample_size_0_val,noise_0 = 0.2,
                                  lower_1 = 1, upper_1 = 2, sample_size_1 = sample_size_1_val, noise_1 = 0.4)
        data.model(w_val)
        figure_1 = data.generate_plot()
        st.plotly_chart(figure_1, use_container_width=True)  
 