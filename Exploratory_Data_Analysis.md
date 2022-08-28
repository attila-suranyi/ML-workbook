# Exploratory Data Analysis
Source: Logistic Regression lecture

### Visualisation
- histograms:
```python
ad_data['Age'].hist(bins=30)
sns.histplot(ad_data,x="Daily Internet Usage",hue="Clicked on Ad")
```
- correlations:
```python
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data)

sns.jointplot(x='Age',y='Daily Time Spent on Site', data=ad_data,kind='kde', color='red');

sns.pairplot(ad_data)
sns.pairplot(ad_data,hue='Clicked on Ad')

sns.heatmap(ad_data.corr())
```

### Missing Data
- drop:
```python
train.drop('Cabin',axis=1,inplace=True)
```
- or fill in, using statistical methods

### Converting Categorical Features
- dummy variable:
```python
sex = pd.get_dummies(train['Sex'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
```

![[Pasted image 20220410151553.png]]