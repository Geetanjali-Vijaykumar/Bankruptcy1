import pandas as pd
from sklearn.naive_bayes import CategoricalNB as CB
from pickle import dump
from pickle import load



dump(model1,open('proj1.sav', 'wb'))

loaded_model=load(open('proj1.sav' ,'rb'))
result = loaded_model.score(X,Y)
print(result)