import torch 
import torch.nn as nn 
import streamlit as st
import numpy as np 
import plotly.graph_objects as go 


#-------------
class LogisticRegression:


    def __init__(self ,lower_0, upper_0, sample_size_0, noise_0,lower_1, upper_1, sample_size_1, noise_1):

        # Parameters
        self.lower_0 = lower_0
        self.upper_0 = upper_0
        self.sample_size_0 = sample_size_0
        self.noise_0 = noise_0
        self.lower_1 = lower_1
        self.upper_1 = upper_1
        self.sample_size_1 = sample_size_1
        self.noise_1 = noise_1

        # made by Paramters
        self.x0 = torch.linspace(lower_0, upper_0, sample_size_0) + torch.tensor([noise_0])
        self.x1 = torch.linspace(lower_1, upper_1, sample_size_1) - torch.tensor([noise_1])
        self.X = torch.cat((self.x0, self.x1), dim=0)
        self.y = torch.cat((torch.zeros(len(self.x0)), torch.ones(len(self.x1))), dim=0)

        # stand alone 
        self.inter_and_extrapolation = torch.linspace(-3,3,1000)
        self.possible_weights = torch.linspace(-5,25,100)
        self.loss_fn = nn.BCEWithLogitsLoss()





    def model(self,w):
        self.w = w 
        L = []

        for weight in self.possible_weights:
            z = weight * self.X
            loss = self.loss_fn(z,self.y)
            L.append(loss)

        L = torch.as_tensor(L)
        secret_weight = self.possible_weights[torch.argmin(L)]

        return secret_weight,L
    

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
            x= self.inter_and_extrapolation,
            y=torch.sigmoid(self.w * self.inter_and_extrapolation),
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
    

    def loss_landscape(self,secret_weight,L):
        
        # landscape
            loss_landscape = go.Scatter(
                    x = self.possible_weights,
                    y = L,
                    mode = 'lines',
                    line = dict(color='pink'),
                    name ='Loss function landscape'
                )

            # global
            Global_minima = go.Scatter(
                x = (secret_weight,),
                y = (torch.min(L),),
                mode = 'markers',
                marker = dict(color='yellow',size=10,symbol='diamond'),
                name = 'Global minima'
            )


            # ball
            z = self.w * self.X     #forward pass
            loss = self.loss_fn(z,self.y)

            ball = go.Scatter(
                    x = (self.w,),
                    y = (loss,),
                    mode = 'markers',
                    marker= dict(color='red'),
                    name = 'loss'
            )

            layout = go.Layout(
                    xaxis = dict(title='w',
                                range = [-8,25],
                                zeroline = True,
                                zerolinewidth = 2,
                                zerolinecolor = 'rgba(205, 200, 193, 0.7)'),

                    yaxis = dict(title='L',
                                range=[0,1.6],
                                zeroline = True,
                                zerolinewidth = 2,
                                zerolinecolor = 'rgba(205, 200, 193, 0.7)')
                )


            figure = go.Figure(data = [loss_landscape,Global_minima,ball],layout=layout)
            
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
        data = LogisticRegression(lower_0 = -3,upper_0 = -1.5, sample_size_0 = sample_size_0_val,noise_0 = 1.2,
                                  lower_1 = 1, upper_1 = 2, sample_size_1 = sample_size_1_val, noise_1 = 1.4)
        data.model(w_val)
        figure_1 = data.generate_plot()
        st.plotly_chart(figure_1, use_container_width=True)  
    

    secret_weight, L = data.model(w_val)

    with col2:
       figure_2 = data.loss_landscape(secret_weight,L)
       st.plotly_chart(figure_2,use_container_width=True)