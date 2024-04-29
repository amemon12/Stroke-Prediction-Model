
import pandas as pd
df=pd.read_csv("C:\\Users\\admin\\Desktop\\healthcare-dataset-stroke-data.csv")
m=df['bmi'].mean()

df['bmi']=df['bmi'].fillna(m)
print(df.isna().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
print("Encoded Classes",le.classes_)
df['ever_married'] = le.fit_transform(df['ever_married'])
print("Encoded Classes",le.classes_)
df['work_type'] = le.fit_transform(df['work_type'])
print("Encoded Classes",le.classes_)
df['Residence_type'] = le.fit_transform(df['Residence_type'])
print("Encoded Classes",le.classes_)
df['smoking_status'] = le.fit_transform(df['smoking_status'])
print("Encoded Classes",le.classes_)
x=df.iloc[:,1:10]
y=df['stroke']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=51)


from sklearn.neighbors import KNeighborsClassifier
kc = KNeighborsClassifier()
kc.fit(x_train,y_train)

from sklearn.metrics import confusion_matrix,accuracy_score
y_pred = kc.predict(x_test)
print("Let's Compare with in % Accuracy")
print("Accuracy score ", accuracy_score(y_test, y_pred.round()))
print("Confusion Matrix", confusion_matrix(y_test, y_pred.round()))

from matplotlib import pyplot as plt
import seaborn as sns
corr=df.corr()['stroke'].sort_values (ascending=False).to_frame()
plt.figure(figsize=(2,8))
sns.heatmap(corr,cmap='Blues',cbar=False,annot=True)
plt.show()

