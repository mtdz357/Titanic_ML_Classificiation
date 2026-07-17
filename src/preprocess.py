import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv(r'C:\Users\Murat\OneDrive - hacettepe.edu.tr\Masaüstü\titanic_project\data\Titanic-Dataset.csv')
# print(train_df.columns)
'''
g = sns.catplot(data=train_df,x='Age',y='Survived',kind='bar')
plt.show()
Age degiskeni oldukca önemli burda bunu model egiitirken kullanicaz

g = sns.catplot(data=train_df,x='Pclass',y='Survived',kind='bar')
plt.show()
Pclass feature i da cok onemli bu da kullanilacak

g = sns.catplot(data=train_df,x='Embarked',y='Survived',kind='bar')
plt.show()
Embarked Feature da kullanilacak

g = sns.catplot(data=train_df,x='Sex',y='Survived',kind='bar')
plt.show()
Cinsiyet survived oranini cok buyuk oranda etkileyen bir feature

g = sns.catplot(data=train_df,x='SibSp',y='Survived',kind='bar')
plt.show()
g = sns.catplot(data=train_df,x='Parch',y='Survived',kind='bar')
plt.show()
Parch ve Sibsp biribirne bagli olan ve aileni buyuklugu ile ilgili olan featurelar

g = sns.catplot(data=train_df,x='Fare',y='Survived',kind='bar')
plt.show()
Fare featurei önemli genelde daha fazla para odeyen insanlarin hayatta kalmasi daha olasi gozukuyor

g = sns.catplot(data=train_df,x='Cabin',y='Survived',kind='bar')
plt.show()
Cabin featurei cok duzenli dagilmis kullanilmamasi daha iyi
'''





# print(train_df.isnull().sum()) # datasetteki eksik ne kadar kisim var bakma
#Cabinde cok fazla eksik veri var ama biz zaten o feature i kullanmicaz o sutunu silebiliriz

TicketList =[]
#Feature Extraction
for i in train_df['Ticket']:
    x=i.split(' ')
    if len(x)==2:
        TicketList.append(x[0])
    else:
        TicketList.append('x')
train_df['T_F'] = TicketList

#Burda kisinin direkt olarak lakabını alarak onun mr mrs master vb olacak sekilde feature cikartiyoruz
NameList=[]
for i in train_df['Name']:
    y = i.split(' ')
    NameList.append(y[1])
train_df['N'] = NameList


#OneHotEncoding for Ticketlist
train_df =pd.get_dummies(train_df,columns=['T_F'],dtype=int)

def title_grupla(title):
    if title in ['Miss.', 'Ms.', 'Mlle.', 'Mrs.']:
        return 'Woman'
    elif title == 'Master.':
        return 'Master'
    elif title == 'Mr.':
        return 'Mr'
    else:
        return 'Others'

train_df['Age'] = train_df.groupby(['N', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)


train_df['N'] = train_df['N'].apply(title_grupla)

train_df = pd.get_dummies(train_df,columns=['N'],dtype=int)
train_df = pd.get_dummies(train_df,columns=['Embarked'],dtype=int)
train_df = pd.get_dummies(train_df,columns=['Pclass'],dtype=int)
train_df.drop(['PassengerId','Sex','Cabin','Name','Ticket'], axis=1,inplace = True)



train_df.to_csv(r'C:\Users\Murat\OneDrive - hacettepe.edu.tr\Masaüstü\titanic_project\data\titanic_data_cleaned.csv', index=False)


def load_split_data(csvpath,targetcolumn,test_size=0.2,random_state=42):
    df =pd.read_csv(csvpath)
    X = df.drop([targetcolumn],axis=1)
    y= df[targetcolumn]

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state)

    scaler = StandardScaler()
    X_train_Scaled = scaler.fit_transform(X_train)
    X_test_Scaled = scaler.transform(X_test)


    return X_train_Scaled,X_test_Scaled,y_train,y_test