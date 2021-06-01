# Bank-Customer-Churn
To know about the project or to view project report please 
<a href="https://github.com/nik-vaibhav18/Bank-Customer-Churn/blob/main/ProjectReport.pdf" target="\_blank"> click here </a>
<br>
<br>
 
Stepwise go through to view the project:

- ***Python Notebooks*** : First go through [Exploratory_Data_Analysis.ipynb](https://github.com/nik-vaibhav18/Bank-Customer-Churn/blob/main/Python%20Notebooks/Exploratory_Data_Analysis.ipynb) to get some insights about the project. Then did the model comparison to get the best suitable model regiowise in [Modelling.ipynb](https://github.com/nik-vaibhav18/Bank-Customer-Churn/blob/main/Python%20Notebooks/Modeling.ipynb) after that build the customer transformers for data preprocessing of the test data before sending into the models for prediction probablity of churn rate for customers in [Custom Transformer.ipynb](https://github.com/nik-vaibhav18/Bank-Customer-Churn/blob/main/Python%20Notebooks/Custom%20Transformer.ipynb). At the end of *Custom Transformer.ipynb* dumped each model region as pickle seperatly for the flask app


- ***Flask Api*** : with the help of dumped custom transformer and models pickle file built the web framework using flask api in [app.py](https://github.com/nik-vaibhav18/Bank-Customer-Churn/blob/main/Flask%20Api/app.py)
