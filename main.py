import torch 
import torch.nn as nn 
import streamlit as st
import numpy as np 
import pandas as pd
import plotly.graph_objects as go 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 

#-------------
class LogisticRegression:


    def __init__(self ,lower_0, upper_0, sample_size_0, noise_0,lower_1, upper_1, sample_size_1, noise_1,w,threshold):

        # Parameters
        self.lower_0 = lower_0
        self.upper_0 = upper_0
        self.sample_size_0 = sample_size_0
        self.noise_0 = noise_0
        self.lower_1 = lower_1
        self.upper_1 = upper_1
        self.sample_size_1 = sample_size_1
        self.noise_1 = noise_1
        self.w = w
        self.threshold = threshold

        # made by attributes
        self.x0 = torch.linspace(lower_0, upper_0, sample_size_0) + torch.tensor([noise_0])
        self.x1 = torch.linspace(lower_1, upper_1, sample_size_1) - torch.tensor([noise_1])
        self.X = torch.cat((self.x0, self.x1), dim=0)
        self.y = torch.cat((torch.zeros(len(self.x0)), torch.ones(len(self.x1))), dim=0)

        # stand alone 
        self.inter_and_extrapolation = torch.linspace(-3,3,1000)
        self.possible_weights = torch.linspace(-5,25,100)
        self.loss_fn = nn.BCEWithLogitsLoss()
#---------------------------------------------------------------
        
# Math for loss_landscape, loss for classes & confusion matrix 

    def Loss(self):
        L = []

        for weight in self.possible_weights:
            z = weight * self.X
            loss = self.loss_fn(z,self.y)
            L.append(loss)

        return torch.as_tensor(L)
    
    # l_class_0 , l_class_1 , l_class_total 
    def loss_per_class(self):
        loss_class_0 = torch.mean(-torch.log(1-torch.sigmoid(self.w * self.x0)))
        loss_class_1 =  torch.mean(-torch.log(torch.sigmoid(self.w * self.x1)))
        loss_class_0_and_1 = (loss_class_0 + loss_class_1)/2

        return loss_class_0,loss_class_1,loss_class_0_and_1
    
    # confusion matrix 
    def make_predictions(self):
        with torch.no_grad():
            prob = torch.sigmoid(self.w * self.X)
            pred = (prob>self.threshold ).float()
            cm = confusion_matrix(self.y,pred,labels=[1,0])
            disp = ConfusionMatrixDisplay(cm,display_labels=['orange','purple'])
            
            disp.plot()

        return plt.gcf()
#---------------------------------------------------------------

    # Data points, sigmoid curve
    def generate_plot(self):
        scatter_class_0 = go.Scatter(
            x=self.x0,
            y=torch.zeros(len(self.x0)),
            mode='markers',
            marker=dict(color='purple'),
            name='class purple'
        )
        scatter_class_1 = go.Scatter(
            x=self.x1,
            y=torch.ones(len(self.x1)),
            mode='markers',
            marker=dict(color='orange'),
            name='class orange'
        )

        non_linear_line = go.Scatter(
            x= self.inter_and_extrapolation,
            y=torch.sigmoid(self.w * self.inter_and_extrapolation),
            mode='lines',
            line={'color': 'rgb(27,158,119)'},
            name='model'
        )


        threshold_line = go.Scatter(
            x = torch.linspace(-10,10,21),
            y = torch.full((21,), self.threshold),
            mode = 'lines',
            line = dict(dash='dash'),
            name = 'Threshold Line'
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
        figure = go.Figure(data=[scatter_class_0, scatter_class_1, non_linear_line,threshold_line], layout=layout)
        return figure
    
#-----------------------------------


    def loss_landscape(self,L):
        
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
                x = (self.possible_weights[torch.argmin(L)],),      # secret weight [The weight that yeilds minimum loss ; Diamond]
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



#------------------------------------------------------
# streamlit 
  

st.set_page_config(layout='wide')
st.title("Logistic Regression : Weight")
st.write('By : Hawar Dzaee')


with st.sidebar:

    st.subheader("Data Generation")
    sample_size_0_val = st.slider("sample size Class 0:", min_value= 2, max_value=12, step=1, value= 7)
    sample_size_1_val = st.slider("sample size Class 1:", min_value= 2, max_value=12, step=1, value= 9) 
    Noise = st.slider('Noise',min_value = 0.0, max_value = 1.0, step = 0.1, value = 0.2)

    st.subheader('Threshold Selection')
    threshold_val = st.slider('threshold',min_value=0.05,max_value=0.99,step=0.05,value=0.5)


    st.subheader("Adjust the parameter(s) to minimize the loss")
    w_val = st.slider("weight (w):", min_value=-4.0, max_value=18.0, step=0.1, value= 1.1) # first and all the widgets above 


container = st.container()


with container:
 

    col1, col2 = st.columns([3,3])

    with col1:

        data = LogisticRegression(lower_0 = -2,upper_0 = 0, sample_size_0 = sample_size_0_val,noise_0 = Noise,
                                  lower_1 = 0, upper_1 = 2, sample_size_1 = sample_size_1_val, noise_1 = Noise,w = w_val,threshold=threshold_val) #second
        data.Loss() # third
        figure_1 = data.generate_plot() # fourth
        st.plotly_chart(figure_1, use_container_width=True)

        st.latex(r'''\hat{{y}} = \frac{1}{1 + e^{-(\color{green}w\color{black}X)}}''')
        st.latex(fr'''\hat{{y}} = \frac{{1}}{{1 + e^{{-(\color{{green}}{{{w_val}}}\color{{black}}X)}}}}''')  

        prob = (torch.sigmoid(data.w * data.X)).tolist()
        prob = [round(i,4) for i in prob]
        
        df = pd.DataFrame({'y\u0302':prob,
                            'y':data.y})


        with st.expander("sigmoid outputs and their corresponding ground truth"):
            st.write(df)

        st.write('-------------')
            

    L = data.Loss() # fifth
    loss_class_0,loss_class_1,loss_class_0_and_1 = data.loss_per_class() # sixth

    with col2:
       figure_2 = data.loss_landscape(L) # seventh
       st.plotly_chart(figure_2,use_container_width=True)
       st.latex(r"""L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]""")
       st.latex(rf"""L_{{\text{{class 0}}}} = \textcolor{{purple}}{{{loss_class_0:.4f}}}  \qquad L_{{\text{{class 1}}}} = \textcolor{{orange}}{{{loss_class_1:.4f}}}""")
       st.latex(rf"""L_{{\text{{total}}}} = \textcolor{{red}}{{{loss_class_0_and_1:.4f}}}""")
       st.write('---------------')
       
       st.subheader('Confusion Matrix On Training Data')
       fig = data.make_predictions() #eighth
       st.pyplot(fig)


st.write('--------------------------')


st.header('Food For Thought')

st.write('''
1. In a real-world scenario, which widget do you have control over?

2. Make the dataset completely separable [Noise = 0]. Is the loss function still convex? Why or why not?

3. When there is no noise [Noise = 0], what is the weight that minimizes the loss function? How is that relevant to the step function?

4. What does maximum noise [Noise = 1] mean? How can you interpret it? Try to find the optimal weight when Noise = 1. What’s your conclusion? 
         
5. Does the threshold affect the loss function? Why or why not? What about the confusion matrix? Why is that?''')






st.write("---")
st.write("Connect with me:")

linkedIn_icon_url = 'https://img.icons8.com/fluent/48/000000/linkedin.png'
github_icon_url = 'https://img.icons8.com/fluent/48/000000/github.png'

html_code = f"""
<div style="display: flex; justify-content: center; align-items: center;">
    <a href="https://www.linkedin.com/in/hawardzaee/" style="margin-right: 10px;">
        <img src="{linkedIn_icon_url}" alt="LinkedIn" style="height: 48px; width: 48px;">
    </a>
    <a href="https://github.com/Hawar-Dzaee" style="margin-left: 10px;">
        <img src="{github_icon_url}" alt="GitHub" style="height: 48px; width: 48px;">
    </a>
</div>
"""

st.markdown(html_code, unsafe_allow_html=True)







