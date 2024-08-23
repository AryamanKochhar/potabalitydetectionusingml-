#Important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style='darkgrid')
import warnings
warnings.filterwarnings('ignore')
import joblib as jb

#Important functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

#Model
from sklearn.tree import DecisionTreeClassifier
#loadinbg the data set
train_df=pd.read_csv('../input/water-potability/water_potability.csv')
train_df.head()
plt.pie(train_df['Potability'].value_counts(),shadow=True, autopct="%f", colors=['aqua', 'red'], radius=2)
plt.legend(['Not Potable','Potable'],loc=(1,1));
#Now we will try to find the relationship of numerical attributes with potability
attributes=[ 'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
            'Organic_carbon', 'Trihalomethanes', 'Turbidity']
used_attribute_list=[]
fig, axes = plt.subplots(3,3, figsize=(20, 20))
k=0
for i in [0, 1, 2]:
    for j in [0,1,2]:
        sns.boxplot(x=train_df.Potability, y=train_df[attributes[k]], ax=axes[i,j])
        k=k+1
plt.figure(figsize=(13,10))
sns.heatmap(train_df.corr())
plt.title('Correlation Between Various Attributes', fontsize=18);
cond=train_df['Potability']==0
#INSTEAD OF REMOVING NAN VALUES IT IS REPLACING IT BY THE MEDIAN VALUES TO THE SIZE OF THE DATA SET REMAINS
train_df['ph'].fillna(cond.map({True:train_df.loc[train_df['Potability']==0]['ph'].median(),
                                False:train_df.loc[train_df['Potability']==1]['ph'].median()
                                }),inplace=True)

train_df['Sulfate'].fillna(cond.map({True:train_df.loc[train_df['Potability']==0]['Sulfate'].median(),
                                False:train_df.loc[train_df['Potability']==1]['Sulfate'].median()
                                }),inplace=True)

train_df['Trihalomethanes'].fillna(cond.map({True:train_df.loc[train_df['Potability']==0]['Trihalomethanes'].median(),
                                False:train_df.loc[train_df['Potability']==1]['Trihalomethanes'].median()
                                }),inplace=True)
