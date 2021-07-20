
import os
os.chdir("C:\\Users\\DDIT SPECIAL CELL\\.spyder-py3\\USModuleXGBoost\\USData Second Month vantage")
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from pandas import DataFrame
#import numpy as np
#from matplotlib import pyplot
#from sklearn.model_selection import train_test_split  
ncol=33
lst_pred=[0]*100
lst_act=[0]*100
datatotdt = pd.read_csv("secondmonth.csv")
datatot=datatotdt.loc[:, datatotdt.columns != "DATE"]
for i in range(0,45):
    data=datatot.loc[0:ncol+i, :]
    X=data.loc[:, data.columns != "GDP Growth QoQ"]
    y=data.loc[:,"GDP Growth QoQ"]
    data_pred=datatot.loc[ncol+i+1:ncol+i+2, data.columns != "GDP Growth QoQ" ]
    #data_pred=datatot.loc[34:79, data.columns != "GDP Growth QoQ" ]
    data_act=datatot.loc[ncol+i+1, "GDP Growth QoQ" ]
    #data_act=datatot.loc[34:79, "GDP Growth QoQ" ]
    model = xgb.XGBRegressor()
    param_search = {'max_depth': range (1, 10, 1),
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.1,0.01,0.3]}
    my_cv = TimeSeriesSplit(n_splits=10)
    #my_cv = [(train,test) for train, test in TimeSeriesSplit(n_splits=10).split(X)]
    gsearch = GridSearchCV(model, cv=my_cv,
                        param_grid=param_search,verbose=True)
    gsearch.fit(X, y)
    preds=gsearch.predict(data_pred)
    print(preds)
    print(data_act)
    lst_pred[i]=preds[0]
    lst_act[i]=data_act
    print(gsearch.best_params_)
    print(gsearch.best_estimator_.feature_importances_)
    
dfile = DataFrame (lst_pred,lst_act)
dfile.to_csv('XGBoostUSresultsSecondmonthvantagewithlagGDP.csv')
