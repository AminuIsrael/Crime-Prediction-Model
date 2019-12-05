import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np

data_2014 = pd.read_csv("Report_2014.csv")
data_2015 = pd.read_csv("Report_2015.csv")
data_2016 = pd.read_csv("Report_2016.csv")
data_2017 = pd.read_csv("Report_2017.csv")
data_2018 = pd.read_csv("Report_2018.csv")

combined = pd.concat([data_2018,data_2017,data_2016,
                      data_2016,data_2015,data_2014],ignore_index=True)

combined = combined.fillna(0)

#Label the location of data we want to predict
combined['Location'] = combined["Location"].map({'AREA-A LION BUILDING':1, 'AREA-B APAPA':2,"AREA-C SURULERE":3,"AREA-D MUSHIN":4,
                                          "AREA-E FESTAC":5,"AREA-F IKEJA":6,"AREA-G OGBA":7,
                                         "AREA-H OGUDU":8,"AREA-J ELEMORO":9,"AREA-K MOROGBO":10,
                                          "AREA-L ILASHE":11,"AREA-M IDIMU":12,"AREA-N IJEDE":13})

combined.drop(["Year"],axis=1,inplace=True)

#Selecting the best 5 features
combined = combined[['CRIME',"Case_Reported","True_case","Location","Adult_F","Adult_M"]]

# Creating the model

#Convert the independent and dependent vairable from a Dataframe to a numpy array
X = combined.iloc[:,1:].values
y = combined.iloc[:,0].values

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval) #Cross validation to check for biasness and variance

# Feature Selection
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(X,y)


#Using the Random forest algorithm to create the predictive model
rf = RandomForestClassifier(n_estimators=50, max_features='sqrt')

# Fitting Model 
model = rf.fit(X,y)

#save model
import pickle

# save the model to disk
model_name  = 'model.pkl'
pickle.dump(model, open(model_name, 'wb'))



# # Making predictions with the model
# no_of_victims_killed = float(input("Number of victims killed(input 0 if there is none): "))
# location = int(input("Please choose location: "))
# adult_m = float(input("Enter number of adult male arrested(input 0 if there is none): "))
# adult_f = float(input("Enter number of adult female arrested(input 0 if there is none): "))
# case_r = float(input("Enter case(s) reported(input 0 if there is none): "))
# case_ref = float(input("Enter case(s) refused(input 0 if there is none): "))
# true_c = float(input("Enter true case(input 0 if there is none): "))
# no_of_victims_GBH= float(input("Number of victims GBH(input 0 if there is none): "))
# year_it_occured = int(input("Year it occured: "))
# age_ma = float(input("Enter number of arrests made with Age 14-17 Males(input 0 if there is none): "))
# age_fa = float(input("Enter number of arrests made with Age 14-17 Females(input 0 if there is none): "))
# U17m = float(input("Enter number of arrests made with under 17 Ages Males(input 0 if there is none): "))
# U17f = float(input("Enter number of arrests made with under 17 Ages Females(input 0 if there is none): "))

# prediction = model.predict([[case_r,case_ref,true_c,adult_m,adult_f,age_ma,age_fa,U17m,U17f
#                             ,no_of_victims_killed,no_of_victims_GBH,location,year_it_occured]])


# prediction = ''.join(prediction)


# print(f"The likely crime to occur is {prediction}")


