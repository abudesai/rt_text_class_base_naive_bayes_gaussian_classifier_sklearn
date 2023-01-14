
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 

from sklearn.naive_bayes import GaussianNB



model_fname = "model.save"
MODEL_NAME = "text_class_base_naive_bayes_sklearn"


class Classifier(): 
    
    def __init__(self, var_smoothing = 1e-1, **kwargs) -> None:
        self.var_smoothing = float(var_smoothing)
        self.model = self.build_model()     
        
        
    def build_model(self): 
        model = GaussianNB(var_smoothing = self.var_smoothing)
        return model
    
    
    def fit(self, train_X, train_y):        
        self.model.fit(train_X, train_y)            
        
    
    def predict(self, X, verbose=False): 
        preds = self.model.predict(X)
        return preds 
    
    
    def predict_proba(self, X, verbose=False): 
        preds = self.model.predict_proba(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)        

    
    def save(self, model_path): 
        joblib.dump(self, os.path.join(model_path, model_fname))        


    @classmethod
    def load(cls, model_path):         
        model = joblib.load(os.path.join(model_path, model_fname))
        return model


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):     
    model = joblib.load(os.path.join(model_path, model_fname))   
    return model





