import os
import pickle
from django.shortcuts import render, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Get the current directory of your Django project's views.py file
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the 'model.pkl' file, relative to your current directory
model_file_path = os.path.join(current_directory, 'model.pkl')

# Load the model using the absolute path
try:
    loaded_model = pickle.load(open(model_file_path, 'rb'))
except FileNotFoundError:
    print(f"Error: 'model.pkl' not found at path: {model_file_path}")

news_csv_path=os.path.join(current_directory,'news.csv')

tfvect=TfidfVectorizer(stop_words='english',max_df=0.7)

dataframe=pd.read_csv(news_csv_path)
x=dataframe['text']
y=dataframe['label']
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# Create your views here.
def home(request):
    return render(request,'index.html')

def fake_news_det(news):
    tfid_x_train=tfvect.fit_transform(x_train)
    tfid_x_test=tfvect.transform(x_test)
    input_data=[news]
    vectorized_input_data=tfvect.transform(input_data)
    prediction=loaded_model.predict(vectorized_input_data)
    return prediction
    

def predict(request):
    if request.method=="POST":
        message=request.POST['news_content']
        prediction=fake_news_det(message)
        print(prediction[0])
        return render(request,'index.html',{'prediction':prediction[0]})
    else:
        return redirect('/')
    