import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings 
warnings.filterwarnings("ignore")

df = pd.read_csv("Heart_Deasis/Heart.csv")

print(df.head())

# First of all Cleaning and  EDA apply on this data

print(df.columns)

print(df.shape)

print(df.info())

print(df.isnull().sum())

print(df.describe())

print(df.duplicated().sum())

(df["HeartDisease"].value_counts().plot(kind = "bar"))
plt.show()



# def plotting1(var , num):
#     plt.subplot(2 , 2 , num)
#     sns.histplot(df[var] , kde=True)
# plotting1("Age" , 1)    
# plotting1("RestingBP" , 2)
# plotting1("Cholesterol" , 3)    
# plotting1("MaxHR" , 4)  

# plt.tight_layout()
# plt.show()



ch_mean = df.loc[df["Cholesterol"] != 0 , "Cholesterol"].mean()
df["Cholesterol"] = df["Cholesterol"].replace(0 , ch_mean)
df["Cholesterol"] = df["Cholesterol"].round(2)
print(ch_mean)


reasting_bp_mean = df.loc[df["RestingBP"] != 0 , "RestingBP"].mean()
df["RestingBP"] = df["RestingBP"].replace(0 , reasting_bp_mean)
df["RestingBP"] = df["RestingBP"].round(2)
print(reasting_bp_mean)
# def plotting(var , num):
#     plt.subplot(2 , 2 , num)
#     sns.histplot(df[var] , kde=True)
    
# plotting("Age" , 1)    
# plotting("RestingBP" , 2)
# plotting("Cholesterol" , 3)    
# plotting("MaxHR" , 4)  

# plt.tight_layout()
# plt.show() 


# sns.countplot(x = df["Sex"] , hue = df["HeartDisease"])
# plt.show()

# sns.countplot(x = df["ChestPainType"] , hue = df["HeartDisease"])
# plt.show()

# sns.countplot(x = df["FastingBS"] , hue = df["HeartDisease"])
# plt.show()

# sns.boxplot(x = "HeartDisease" , y = "Cholesterol" , data = df)
# plt.show()

# sns.violinplot(x = "HeartDisease" , y = "Age" , data = df)
# plt.show()

# sns.heatmap(df.corr(numeric_only = True) , annot = True)
# plt.show()


# DATA PREPROCESSING AND CLEANING

df_encode = pd.get_dummies(df , drop_first = True)
print(df_encode)

df_encode =df_encode.astype(int)
print(df_encode)

numerical_cols = ["Age" , "RestingBP" , "Cholesterol" , "MaxHR" , "Oldpeak" ]
scalar = StandardScaler()
df_encode[numerical_cols] = scalar.fit_transform(df_encode[numerical_cols])
print(df_encode.head())

print(df_encode.columns)

X = df_encode.drop("HeartDisease" , axis=1)
y = df_encode["HeartDisease"]

print(X)


X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.20, random_state=42)

scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.fit_transform(X_test)


model = {
    "Logistic Regression" : LogisticRegression(),
    "KNN" : KNeighborsClassifier(),
    "naive_bayes" : GaussianNB(),
    "Decision Tree" : DecisionTreeClassifier(),
    "SVM" : SVC() 
}

result = []

for name , clf in model.items():
    clf.fit(X_train_scaled , y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test , y_pred)
    f1 = f1_score(y_test , y_pred)
    
    result.append({
        "model" : name,
        "Accuracy" : round(acc , 4),
        "f1 score" : round(f1 , 4)
    })
    
print(result)

import joblib

joblib.dump(model["KNN"] , "KNN_heart.pkl")
joblib.dump(scalar , "scalar.pkl")
joblib.dump(X.columns.tolist() , "columns.pkl")
print("Fill successfully saved")


