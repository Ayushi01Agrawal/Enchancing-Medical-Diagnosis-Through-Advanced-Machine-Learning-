import streamlit as st
import pandas as pd
import numpy as np

import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression

from streamlit_lottie import st_lottie

st.title(":blue[Cardiovascular heart disease data assesment]")

heart_df=pd.read_csv("CHD_preprocessed.csv")
heart_df.drop(['education'],axis=1,inplace=True)

X = heart_df.drop(columns = 'TenYearCHD', axis=1)
Y = heart_df['TenYearCHD']

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle = True, test_size = .2, random_state = 44)

def evaluate_model(model, x_test, y_test):
    from sklearn import metrics

    # Predict Test Data
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa,
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}

clf = tree.DecisionTreeClassifier(random_state=0)
clf.fit(X_train, Y_train)

# Evaluate Model
clf_eval = evaluate_model(clf, X_test, Y_test)


def ValueCount(str):
  if str=="Yes":
    return 1
  else:
    return 0
def Sex(str):
  if str=="Male":
    return 1
  else:
    return 0


d1=[" ","Female","Male"]
val1 = st.selectbox("Gender  ",d1)
val1 = Sex(val1)

val2 = st.number_input('Age')

d2=[" ","Yes","No"]
val3 = st.selectbox("Do you smoke?  ",d2)
val3 = ValueCount(val3)

val4 = st.number_input('How many cigarette do you have per day?')

val5 = st.selectbox("Do you take medicine for Blood Pressure?  ",d2)
val5 = ValueCount(val5)

val6 = st.selectbox("Did you have stroke in past?  ",d2)
val6 = ValueCount(val6)

val7 = st.selectbox("Are you hypertensive?  ",d2)
val7 = ValueCount(val7)

val8 = st.selectbox("Do you have diabetes?  ",d2)
val8 = ValueCount(val8)

val9 = st.number_input('Cholesterol level ')

val10 = st.number_input('Systolic blood pressure level ')

val11 = st.number_input('Ciastolic blood pressure level ')

val12 = st.number_input('Body Mass Index ')

val13 = st.number_input('Heart Rate ')

val14 = st.number_input('Glucose level ')

# input_data = (0	,50,	0,	0,	0,	0,	0,	0,	254,	133,	76,	22.91,	75,	76)
# changing the input_data to numpy array
input_data = (val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val11,val12,val13,val14)

input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
# print(std_data)

prediction = clf.predict(std_data)
# print(prediction)
with st.expander("Analyze provided data"):
  st.subheader("Results:")

  if (prediction[0] == 0):
    st.info('The person may not have heart disease.')
  else:
    st.warning('The person may have heart disease')