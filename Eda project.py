# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 18:31:20 2025

@author: harpr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel(r"C:\Users\harpr\Downloads\June_2024_May_2025_Transaction (2).xlsx")

#check the shape of data
data.shape
#check the datatype
data.dtypes
#Exploratory data analysis
#Discriptive analysis

data.describe()

# we have to convert the the object to numeric which is mixed

col_to_convert = ['Quantity', 'Rate', 'Item Grams', 'Item Net Weight', 'Minimum Order', 'Invoice Quantity']

for col in col_to_convert:
    data[col] = data[col].astype(str).str.replace(',', '').str.extract(r'(\d+\.?\d*)')
    data[col] = pd.to_numeric(data[col], errors='coerce') 
    
numeric_col = data.select_dtypes(include = ['int64', 'float64']).columns 

data[numeric_col].info()

#seperate date column so easily analyse
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Invoice Date'] = pd.to_datetime(data['Invoice Date'], errors='coerce')
data['Order Released Date'] = pd.to_datetime(data['Order Released Date'], errors='coerce')

data['Factory Dispatch (Parsed)'] = pd.to_datetime(data['Factory Dispatch'], errors='coerce')

# extratr month from date time
data['order month'] = data['Date'].dt.month
data['Invoice month'] = data['Invoice Date'].dt.month
data['Release month'] = data['Order Released Date'].dt.month

# make a sperate column to list of date
date_col = data.select_dtypes(include = ['datetime64']).columns

# Category column

categ_col = data.select_dtypes(include = ['object']).columns

# Now we do univariate anlysis
#Hist plot
for col in numeric_col:
    plt.figure(figsize = (8,4))
    sns.histplot(data[col], kde = True, color = 'skyblue')
    plt.title(f"Histogram of {col}")
    plt.grid(True)
    plt.show()
    

#Bar plot

for col in categ_col:
    plt.figure(figsize = (8,4))
    data[col].value_counts().head(10).plot(kind = 'bar', color = 'orange')
    plt.title(f"Top categories in {col}")
    plt.grid(True)
    plt.show()
    
    
#Box Plot

for col in numeric_col:
    plt.figure(figsize = (8,4))
    sns.boxplot(x= data[col], color = 'lightgreen' )
    plt.title(f"boxplot of {col}")
    plt.grid(True)
    plt.show()  
    
# or alternative    
    
data[numeric_col].plot(kind = 'box', subplots = True, sharey = False , figsize = (8,4))
plt.show()  

#Count plot of categorical column
for col in categ_col:
    plt.figure(figsize = (8,4))
    sns.countplot(x = col, data = data , color = 'blue')
    plt.title(f"countplot of {col}")
    plt.grid(True)
    plt.show()


#Box plot for categorical data
#for col in categ_col:
    #print(f"\n🔹 Column: {col}")
   # print(data[col].value_counts())
    
    
#for col in categ_col:
   # print(f"\n🔹 Column: {col}")
   # print((data[col].value_counts(normalize=True) * 100).round(2))   
   
   
#Bivariate analysis:    Comparing 2 variable togeather to understand relationship

#Numerical VS Numerical

#Correlation Heatmap

plt.figure(figsize = (8, 6))
sns.heatmap(data[numeric_col].corr(), annot = True , cmap = 'coolwarm')
plt.title('Correlation map')
plt.show()      


#Scatter plot
sns.scatterplot(x = 'Quantity', y = 'Rate', data = data)  
plt.title('Quantity VS Rate')
plt.show()
 

sns.scatterplot(x = 'Rate', y = 'Item Grams', data = data)
plt.title('rate vs item gram')
plt.show()

 

sns.scatterplot(x = 'Minimum Order', y = 'Rate', data = data)
plt.title('Minimum order vs Rate')
plt.show()  


correlation = data['Minimum Order'].corr(data['Rate'])
print(f"Correlation between Minimum Order and Rate: {correlation:.2f}")



#Categorial VS Numerical

sns.boxplot(x = 'Authorization Status', y = 'Quantity', data = data)
plt.title('Authorization status Vs Quantity')
plt.xticks(rotation = 45)
plt.show()


#barplot

sns.barplot(x = 'Item Category', y = 'Quantity', data = data , estimator = 'mean')
plt.title('Average quantity by Item category')
plt.show()

#Categorail VS categorical

sns.countplot(x = 'Authorization Status', hue = 'Item Category', data = data)
plt.xticks(rotation = 45)
plt.title('Authorization vs item category')
plt.show()

#Cross tab

ct = pd.crosstab(data['Authorization Status'], data['Sourcing Type'])
sns.heatmap(ct, annot=True, cmap='Blues')
plt.title("Crosstab: Authorization Status vs Sourcing Type")
plt.show()


for col in numeric_col:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Authorization Status', y=col, data=data)
    plt.title(f'{col} by Authorization Status')
    plt.xticks(rotation=45)
    plt.show()
    

sns.countplot(x = 'Order Status', hue = 'Item Category', data = data)
plt.title('Order status vs item category')
plt.show()    
    
    
    
    