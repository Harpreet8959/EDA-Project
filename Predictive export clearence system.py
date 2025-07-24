# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 18:26:37 2025

@author: harpr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sqlalchemy import create_engine
from urllib.parse import quote
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sklearn.metrics as skmet 
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from feature_engine.outliers import Winsorizer
data =  pd.read_excel(r"C:\Users\harpr\Downloads\June_2024_May_2025_Transaction (2).xlsx")

user = 'root'
pw = quote("Mansi@11")
db = "Student_DB"
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

data.to_sql('export_data', con=engine, if_exists = 'replace', index = False)

sql = 'select * from export_data;'

export_data = pd.read_sql_query(sql , engine.connect())


export_data.shape

export_data.columns

export_data.info()

export_data.describe()

export_data.isnull().sum()
export_data.duplicated().sum()

export_data = export_data.drop_duplicates()

#Clean columns name
export_data.columns = export_data.columns.str.strip().str.lower().str.replace(' ', '_')

#Clean and set target column

def clean_order_status(status):
    if pd.isna(status) or status in ['Unknown', 'None']:
        return 'Unknown'
    elif status in ['Order Pending', 'Pending']:
        return 'Pending'
    elif status in ['Order Released']:
        return 'Released'
    elif status == 'Approved':
        return 'Approved'
    else:
        return status
export_data['order_status_clean'] = export_data['order_status'].apply(clean_order_status)
export_data = export_data[export_data['order_status_clean'] != 'Unknown']  

#drop the order status 
export_data.drop(columns = ['order_status'], inplace = True)

#Handle unknown mising value na

export_data.replace(['Unknown','unknown', 'NA', 'na', 'N/A', 'Nan', 'nan', 'NaT', None],np.nan , inplace = True)

#Check data types and convert

export_data.dtypes

#Convert object to float

for col in export_data.columns:
    if export_data[col].dtype == 'object':
        try:
            export_data[col] = pd.to_numeric(export_data[col])
        except:
            continue
        
#Check again datatypes
export_data.dtypes        

#Now convert date columns

date_col = ['invoice_date', 'order_released_date']

for col in date_col:
    export_data[col] = pd.to_datetime(export_data[col], errors = 'coerce')
    
    
#Check how many values are missing in datetime collumn
print(export_data['order_released_date'].isna().sum())    
print(export_data['invoice_date'].isna().sum())

export_data.drop(columns = ['invoice_date'], inplace = True)

#fill order released date with median and add flag so model wont confused

export_data['missing_order_date'] = export_data['order_released_date'].isna().astype(int)
median_order_date = export_data['order_released_date'].median()
export_data['order_released_date'] = export_data['order_released_date'].fillna(median_order_date) 



#Drop irrelavant columns
nulls = export_data.isnull().mean()
export_data = export_data.drop(columns = nulls[nulls > 0.5].index)  

# drop identifiers

export_data.drop(columns=[
    'serial_number', 'document_number', 'invoice_number', 'order_number',
    'note_1', 'note_2', 'extra_fields', 'extra_fields_final_destination'
], errors='ignore', inplace=True)

#check the columns and shape of data
export_data.columns

#drop irelavant columns
columns_to_drop = [
    'create_new_user', 'customer_account_name', 'customer_po_number',
    'item_name', 'capital_asm_employee_name', 'consignee',
    'narration'
]
export_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

#check the columns and shape of data
export_data.columns


#Now extract year/month from date

#Convert the date to datetime

export_data['date'] = pd.to_datetime(export_data['date'], errors = 'coerce')

# Step 2: Drop rows where date couldn't be parsed (optional — or you can fill later)
export_data = export_data.dropna(subset=['date'])

#Extract month and year from date
export_data['order_month'] = export_data['date'].dt.month
export_data['order_year'] = export_data['date'].dt.year

# drop original date column
export_data.drop(columns = ['date'], inplace = True)

export_data.columns

#first convert categorical to numerical


#feature engineering by domain knowledge

export_data['total_weight'] = export_data['item_net_weight'] * export_data['quantity'] 


#create new feature from minimum order

import re

export_data['moq_count'] = export_data['minimum_order'].str.extract(r'(\d+)').astype(float)
export_data['moq_container'] = export_data['minimum_order'].str.extract(r'([a-zA-Z]+\d+[A-Z]+)', expand=False)

#drop the original column
export_data.drop(columns = ['minimum_order'], inplace = True)

#Now detect the target column and numerical column and categorical column
export_data.dtypes

target = 'order_status_clean'
num_col = export_data.select_dtypes(include = ['int32', 'float64']).columns.tolist()
num_col = [col for col in num_col if col != target]


#Fill missing value wiht median
export_data[num_col] = export_data[num_col].fillna(export_data[num_col].median())


#check the outliers in numeric column by boxplot
export_data[num_col].plot(kind = 'box', subplots = True , sharey = False, figsize = (8,6))
plt.show()

# Treate the outlier using winsorization'

winsor = Winsorizer(capping_method = 'iqr',
                    tail = 'both',
                    fold = 1.5,
                    variables = num_col
                    )
#Now fit the winsor to data num col
export_data[num_col] = winsor.fit_transform(export_data[num_col])

#Check the ouutlier using boxplot

export_data[num_col].plot(kind = 'box', subplots = True , sharey = False, figsize = (8,6))
plt.show()

#Scale the num columns

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
export_data[num_col] = scaler.fit_transform(export_data[num_col])



#Categorical columns
cat_col = export_data.select_dtypes(include = ['object']).columns.tolist()
cat_col = [col for col in cat_col if col != target]


#impute Categorical columns by mode

for col in cat_col:
    if export_data[col].isnull().sum() > 0:
        mode_val = export_data[col].mode()[0]
        export_data[col].fillna(mode_val, inplace=True)



#Check uniqness of categorical column
export_data[cat_col].nunique()

#Group High carditionality column
top_countries = export_data['country_of_fd_name'].value_counts().nlargest(10).index
export_data['country_of_fd_name'] = export_data['country_of_fd_name'].apply(lambda x: x if x in top_countries else 'Other')

item_logic = export_data['item_location_name'].value_counts().nlargest(10).index
export_data['item_location_name'] = export_data['item_location_name'].apply(lambda x: x if x in item_logic else 'Other')

port_of_discharge = export_data['port_of_discharge'].value_counts().nlargest(10).index
export_data['port_of_discharge'] = export_data['port_of_discharge'].apply(lambda x: x if x in port_of_discharge else 'Other')

port_of_loading = export_data['port_of_loading'].value_counts().nlargest(10).index
export_data['port_of_loading'] = export_data['port_of_loading'].apply(lambda x: x if x in port_of_loading else 'Other')



#Encode column
from sklearn.preprocessing import LabelEncoder
#Label encode binary column

le = LabelEncoder()
export_data['sourcing_type'] = le.fit_transform(export_data['sourcing_type'])
export_data['moq_container'] = le.fit_transform(export_data['moq_container'])    

# One-hot encode nominal columns
cat_onehot = ['authorization_status', 'item_category', 'item_packet_size', 
              'invoice_status', 'country_of_fd_name', 'item_location_name',
              'port_of_loading', 'port_of_discharge']  # After grouping high-card ones

export_data = pd.get_dummies(export_data, columns=cat_onehot, drop_first=True)

#Drop high carditinality column
export_data = export_data.drop('factory_dispatch', axis=1)


#Now handle target columnn
#check the value count
export_data['order_status_clean'].value_counts()
export_data['order_status_clean'].value_counts(normalize = True)


#Encode the target column
le = LabelEncoder()
export_data['order_status_clean'] = le.fit_transform(export_data['order_status_clean'])


#Now split the target and input feature

# Identify datetime columns
datetime_cols = export_data.select_dtypes(include=['datetime64[ns]']).columns

# Drop them for now
X = export_data.drop(columns=['order_status_clean'] + list(datetime_cols))
y = export_data['order_status_clean']

'''#Now split the data
X_train , X_test, y_train , y_test = train_test_split(X,y , test_size = 0.2, stratify = y , random_state = 42)


#Now apply smote for balancing target column
#from imblearn.over_sampling import SMOTE

#smote = SMOTE(random_state=42)
#X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

#Check the balance class
#pd.Series(y_train_smote).value_counts()

#Now train the model
#Decision Tree
#from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(class_weight = 'balanced', random_state = 42)
dt_model.fit(X_train, y_train)

#Now predict the model on train data
pred_train_dt = dt_model.predict(X_train)

#Now made confusion matrix
pd.crosstab(y_train , pred_train_dt , rownames = ['Actaul'], colnames = ['Prediction'])

print(skmet.accuracy_score(y_train , pred_train_dt))

#Now test it on test data

pred_test_dt = dt_model.predict(X_test)

pd.crosstab(y_test , pred_test_dt, rownames = ['Actual'], colnames=['Prediction'])

print(accuracy_score(y_test , pred_test_dt))

#Classification report

print("Decision Tree:\n", classification_report(y_test, pred_test_dt, target_names=le.classes_))


#Tune the decision tree hyperparameter tuning

dt_tune = DecisionTreeClassifier(
    class_weight = 'balanced',
    max_depth = 10,
    min_samples_split = 50,
    min_samples_leaf = 20,
    random_state = 42
  )

dt_tune.fit(X_train , y_train)  

pred_test_dt_tune = dt_tune.predict(X_test)        

#Confusion matrix

pd.crosstab(y_test , pred_test_dt_tune , rownames = ['Actual'], colnames = ['Prediction'])

print(accuracy_score(y_test , pred_test_dt_tune))

print("Decision Tree:\n", classification_report(y_test, pred_test_dt_tune, target_names=le.classes_))




###Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators = 100,
                                  max_depth = 10,
                                  min_samples_leaf = 5,
                                  class_weight = 'balanced',
                                  random_state = 42
                                 ) 
#fit the model on train data
rf_model.fit(X_train, y_train)

#predict on train data
pred_train_rf = rf_model.predict(X_train)

#Confusion matrix
pd.crosstab(y_train, pred_train_rf , rownames = ['Actual'], colnames=['Prediction'])

#print accuracy score
print(accuracy_score(y_train , pred_train_rf))

#Now predit on test data

pred_test_rf = rf_model.predict(X_test)
#Confusion matrix
pd.crosstab(y_test , pred_test_rf, rownames = ['Actaul'], colnames = ['Prediction'])

#Print accuracy
print(accuracy_score(y_test , pred_test_rf))

#Evaluation

print("Random Forest:\n", classification_report(y_test, pred_test_rf, target_names=le.classes_))

##try balancerandom forest
from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(
    n_estimators = 100,
    max_depth = 10,
    min_samples_leaf = 5,
    random_state = 42
    )
    
brf.fit(X_train , y_train)
pred_test_brf = brf.predict(X_test)

#confusion matrix

pd.crosstab(y_test, pred_test_brf , rownames=['Actual'], colnames=['Prediction'])

print(classification_report(y_test, pred_test_brf, target_names=le.classes_))



####XGBoost model

from xgboost import XGBClassifier

xgb = XGBClassifier(
    objective = 'multi:softmax',
    num_class = 3,
    eval_metric = 'mlogloss', 
    use_label_encoder = False,
    n_estimators = 200,
    scale_pos_weight = 2,
    random_state = 42
    )

xgb.fit(X_train , y_train)

#Now predict on train data

pred_train_xgb = xgb.predict(X_train)

#Confusion matrix

pd.crosstab(y_train, pred_train_xgb , rownames=['Actual'], colnames=['Prediction'])

print(classification_report(y_train, pred_train_xgb, target_names=le.classes_))

#Now predict on test data

pred_test_xgb = xgb.predict(X_test)

##Confuison matrix

pd.crosstab(y_test, pred_test_xgb , rownames=['Actual'], colnames=['Prediction'])

print(classification_report(y_test, pred_test_xgb, target_names=le.classes_))


#Check features importace
import matplotlib.pyplot as plt
from xgboost import plot_importance
plot_importance(xgb)
plt.show()




####Compare these three model

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Store model names
model_names = ['Decision Tree', 'Random Forest', 'Balanced Random Forest', 'XGBoost']
models = [dt_model, rf_model, brf, xgb]  # Make sure these variables are your trained models

# Initialize empty list to store results
results = []

# Loop over each model and collect performance
for name, model in zip(model_names, models):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append([name, acc, precision, recall, f1])

# Create dataframe for comparison
import pandas as pd
comparison_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
comparison_df = comparison_df.sort_values(by='F1 Score', ascending=False)
print(comparison_df)
'''

from sklearn.model_selection import train_test_split

X_temp , X_test , y_temp , y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

#Split the dataset in train validation

X_train ,X_val , y_train , y_val,  = train_test_split(X_temp , y_temp, test_size = 0.2,stratify= y_temp, random_state = 42)


##Use smote for balance

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#Now fit the base model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)

#fit the model in train data
rf.fit(X_train_resampled , y_train_resampled)

#predict on validation set
val_pred = rf.predict(X_val)

#confusion matrix
pd.crosstab(y_val , val_pred , rownames=['Actual'], colnames=['Prediction'])

#classification reprot

from sklearn.metrics import classification_report


print(classification_report(y_val, val_pred, target_names=le.classes_))


from collections import Counter
print(Counter(y_train_resampled))


#Tune the hyperparameter of random forest

import numpy as np

# Get probabilities
proba = rf.predict_proba(X_val)

custom_pred = []
for p in proba:
    if p[0] > 0.2:  # 20% threshold for Approved (be more sensitive)
        custom_pred.append(0)
    elif p[1] > 0.5:
        custom_pred.append(1)
    else:
        custom_pred.append(2)

# Evaluate
print(classification_report(y_val, custom_pred, target_names=le.classes_))



