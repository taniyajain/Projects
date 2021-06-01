import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


class model_build(BaseEstimator, TransformerMixin):
    
    def __init__(self,country):
        if country=='France':
            self.params = {'num_leaves': 5,
                         'n_estimators': 138,
                         'min_child_samples': 60,
                         'max_depth': 3,
                         'learning_rate': 0.03}
        
        elif country=='Germany':
            self.params = {'num_leaves': 11,
                         'n_estimators': 185,
                         'min_child_samples': 55,
                         'max_depth': 2,
                         'learning_rate': 0.05}
        
        elif country=='Spain':
            self.params = {'num_leaves': 10,
                         'n_estimators': 121,
                         'min_child_samples': 30,
                         'max_depth': 14,
                         'learning_rate': 0.03}
        
        elif country=='NK':
            self.params = {'num_leaves': 38,
                         'n_estimators': 113,
                         'min_child_samples': 72,
                         'max_depth': 3,
                         'learning_rate': 0.09}
        else:
            print('Wrong Country!!')
        
        self.lgbc = LGBMClassifier(**self.params)
        
    def fit(self,X,y):
        self.lgbc.fit(X,y)
        return self
        
        
    def predict(self,X):
        if X.shape[0] > 0:
            ypred = self.lgbc.predict(X)
            
            return ypred
    
    def predict_proba(self,X):
        if X.shape[0]>0:
            yprob = self.lgbc.predict_proba(X)
            return yprob


class data_prep_1(BaseEstimator, TransformerMixin):
    def __init__(self,fill_value='?'):
        self.fill_value = fill_value
    def fit(self,df):
        return self
    def transform(self,df):
        df_gender_null_counter = df['Gender'].isnull().sum()
        if df_gender_null_counter>0:
            df_isgender_null = True
            df['Gender'].fillna(value = self.fill_value)
        return (df,df_gender_null_counter)

    
class data_prep_2(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self,df,df_gender_null_counter):
        return self
    def transform(self,df,df_gender_null_counter=0):
        le = LabelEncoder()
        df.loc[:,'Gender'] = le.fit_transform(df['Gender'])
        if df_gender_null_counter>0:
            df.loc[:,'Gender'] = df['Gender'].replace({0:np.nan})
        return df

    
class data_prep_3(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self,X=None):
        return self
    
    def transform(self,train,X=None):
        
        trainF = train[train['Geography']=='France']
        

        trainG = train[train['Geography']=='Germany']
        

        trainS = train[train['Geography']=='Spain']
        
        trainNK = train.copy()
        
        trainF.drop(columns=['Geography'], inplace=True)
        trainG.drop(columns=['Geography'], inplace=True)
        trainS.drop(columns=['Geography'], inplace=True)
        trainNK.drop(columns=['Geography'], inplace=True)    
        return (trainF,trainG,trainS,trainNK)


class data_prep_4(BaseEstimator, TransformerMixin):
    
    def __init__(self,num_bool):
        self.num_bool=num_bool
        
    
    def fit(self,X,y=None):
        if self.num_bool==True:
            self.est = RandomForestRegressor()
            self.itimp = IterativeImputer(self.est)
            self.cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                          'EstimatedSalary']
            
        else:
            self.est = RandomForestClassifier()
            self.itimp = IterativeImputer(self.est)
            self.cols =  ['Gender','HasCrCard','IsActiveMember']
            
        self.itimp.fit(X[self.cols])
        
        
        return self
        
    def transform(self,X,y=None):
        if X.shape[0] > 0:
            X.loc[:, self.cols] = self.itimp.transform(X[self.cols])
           
            return X
        else:
            return X

class data_prep_5(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self,df):
        self.cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                          'EstimatedSalary']  
        self.pt = PowerTransformer()
        self.pt.fit(df[self.cols])
        return self
        
    def transform(self,df):
        if df.shape[0] > 0:
            df.loc[:,self.cols] = self.pt.transform(df[self.cols])
            return df

class Combined_Data_Prep(BaseEstimator, TransformerMixin):
    
    def __init__(self):

        self.test_bool = False
        self.country = str
       
        self.d1 = data_prep_1()
        self.d2 = data_prep_2()
        self.d3 = data_prep_3()
        
    
        self.d41F = data_prep_4(num_bool=True)
        self.d42F = data_prep_4(num_bool=False)
        
        self.d41G = data_prep_4(num_bool=True)
        self.d42G = data_prep_4(num_bool=False)
        
        self.d41S = data_prep_4(num_bool=True)
        self.d42S = data_prep_4(num_bool=False)
        
        self.d41NK = data_prep_4(num_bool=True)
        self.d42NK = data_prep_4(num_bool=False)
        
        self.d5F = data_prep_5()
        self.d5G = data_prep_5()
        self.d5S = data_prep_5()
        self.d5NK = data_prep_5()
        
    def fit(self,X,y=None):
        return self
        
        
    def transform(self,X,y=None):
        
        if self.test_bool==False:
            
            train,b = self.d1.fit_transform(X)
            
            train = self.d2.fit_transform(train,df_gender_null_counter=b)

            trainF,trainG,trainS,trainNK= self.d3.transform(train)

            trainF = self.d41F.fit_transform(trainF)        
            trainF = self.d42F.fit_transform(trainF)

            trainG = self.d41G.fit_transform(trainG)
            trainG = self.d42G.fit_transform(trainG)

            trainS = self.d41S.fit_transform(trainS)
            trainS = self.d42S.fit_transform(trainS)

            trainNK = self.d41NK.fit_transform(trainNK)
            trainNK = self.d42NK.fit_transform(trainNK)
            
            trainF = self.d5F.fit_transform(trainF)
            trainG = self.d5G.fit_transform(trainG)
            trainS = self.d5S.fit_transform(trainS)
            trainNK = self.d5NK.fit_transform(trainNK)

            return (trainF,trainG,trainS,trainNK)

        else:
            
            dptest,c = self.d1.fit_transform(X)
            
            dptest = self.d2.fit_transform(dptest,df_gender_null_counter=c)

            dptest.drop(columns=['Geography'],inplace=True)
            if self.country=='France':
                dptest = self.d41F.transform(dptest)
                dptest = self.d42F.transform(dptest)
                dptest = self.d5F.transform(dptest)
                return dptest
                
            elif self.country=='Germany':
                dptest = self.d41G.transform(dptest)
                dptest = self.d42G.transform(dptest)
                dptest = self.d5G.transform(dptest)
                return dptest

            elif self.country=='Spain':
                dptest = self.d41S.transform(dptest)
                dptest = self.d42S.transform(dptest)
                dptest = self.d5S.transform(dptest)
                return dptest

            elif self.country=='NK':
                dptest = self.d41NK.transform(dptest)
                dptest = self.d42NK.transform(dptest)
                dptest = self.d5NK.transform(dptest)
                return dptest
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    