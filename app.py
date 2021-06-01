from flask import Flask, render_template, request
import jsonify
import requests
import pickle

import pandas as pd
import numpy as np
import sklearn

from data_prep import Combined_Data_Prep,data_prep_1, data_prep_2, data_prep_3, data_prep_4, data_prep_5
from data_prep import model_build


app = Flask(__name__)
data_prep = pickle.load(open('data_processing_custom_transformer.pkl','rb'))



modelF = pickle.load(open('modelF.pkl','rb'))
modelG = pickle.load(open('modelG.pkl','rb'))
modelS = pickle.load(open('modelS.pkl','rb'))
modelNK = pickle.load(open('modelNK.pkl','rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html.txt')


@app.route("/predict", methods=['POST'])
def predict():
    
        try:



        
            country = request.form['Country']
            if country=='Not_Known':
                country='NK'
            
            gender = request.form['Gender']
            if gender == 'Male':
                gender = 1
            elif gender == 'Female':
                gender = 0
            elif gender=='Not Known':
                gender = np.nan
        
        
            isactmem = request.form['IsActiveMember']
            if isactmem=='Yes':
                isactmem=1
            elif isactmem=='No':
                isactmem=0
            else:
                isactmem = np.nan
        
            hascrcard = request.form['HasCrCard']
            if hascrcard=='Yes':
                hascrcard=1
            elif hascrcard == 'No' :
                hascrcard = 0
            else:
                hascrcard = np.nan
        
            creditscore = request.form['Credit Score']
            if creditscore=='':
                creditscore = np.nan
            else:
                creditscore = int(creditscore)
        
            Age = request.form['Age']
            if Age=='':
                Age=np.nan
            else:
                Age = int(Age)
        
            Tenure = request.form['Tenure']
            if Tenure=='':
                Tenure=np.nan
            else:
                Tenure = int(Tenure)
            
        
            Balance = request.form['Balance']
            if Balance=='':
                Balance = np.nan
            else:
                Balance = float(Balance)
        
            Number_of_Products = request.form['Number of Products']
            if Number_of_Products=='':
                Number_of_Products=np.nan
            else:
                Number_of_Products=int(Number_of_Products)
        
            Estimated_Salary = request.form['Estimated Salary']
            if Estimated_Salary=='':
                Estimated_Salary = np.nan
            else:
                Estimated_Salary = float(Estimated_Salary)
            cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance','NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
       
        
            deployed_test = pd.DataFrame([[creditscore,country,gender,Age,Tenure,Balance,Number_of_Products,hascrcard,isactmem,Estimated_Salary]], columns=cols)
        
            data_prep.test_bool = True
            data_prep.country=country
            deployed_test = data_prep.fit_transform(X=deployed_test)
        
            if country=='France':
                ypred = modelF.predict(deployed_test)
                yprob = modelF.predict_proba(deployed_test)[:,1]
        
            elif country=='Germany':
                ypred = modelG.predict(deployed_test)
                yprob = modelG.predict_proba(deployed_test)[:,1]
        
            elif country=='Spain':
                ypred = modelS.predict(deployed_test)
                yprob = modelS.predict_proba(deployed_test)[:,1]
        
            elif country=='NK':
                ypred = modelNK.predict(deployed_test)
                yprob = modelNK.predict_proba(deployed_test)[:,1]
        
            if ypred==1:
                output = 'The customer is likely to churn (probability of churn = ' + str(round(yprob[0]*100, 2)) + '%)'
        
            else:
                output = 'Customer is likely to be retained (probability of retention = ' + str(round((1-yprob[0])*100, 2)) + '%)'
        
        
            return render_template('result.html.txt',prediction_text=output)

        except Exception as e:
            print('The Exception message is: ', e)
            return render_template('result.html.txt',prediction_text='Invalid Input')

    else:
        return render_template('index.html')




if __name__=="__main__":
    app.run(debug=True)