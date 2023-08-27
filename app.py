#import imports
from flask import Flask, request, app, render_template, Response
import pandas as pd
import numpy as np
import pickle

application= Flask(__name__)
app = application

#loading pickle model
scaler = pickle.load(open('Model/StandardScaler.pk1'))
model = pickle.load(open('Model/modelForPredictingDiabetes.pk1'))

#routing to homepage
@app.route('/')
def index():
    return render_template('index.html')

##route for single datapoint prediction
@app.route('prediction',methods = ['GET','POST'])
def pred_datapoints():
    result = ""
    if request.method=='POST':
        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
        
        new_data = scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        prediction = model.predict(new_data)
        
        if prediction[0]==1:
            result = 'Diabetic'
        else:
            result = "Non-Diabetic"
            
        return render_template('single_prediction.html',result=result)
    else:
        return render_template('home.html')
if __name__=="__main__":
    app.run(host="0.0.0.0")