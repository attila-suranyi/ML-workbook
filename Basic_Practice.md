## Exploratory Data Analysis

### Visualisation
- Histograms:
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
- It is beneficial if there isn't a strong connection between our features, because that means there isn't much redundancy.

### Missing Data
- Drop:
```python
train.drop('Cabin',axis=1,inplace=True)
```
- or fill in: using the most frequent value, the mean, or some statistical method

### Converting Categorical Features
- Dummy variable:
```python
sex = pd.get_dummies(train['Sex'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
```
- When converting to numerical features, and the order doesn't have any additional meaning, avoid simply creating 0, 1, 2... numerical values, because that would imply to the model that order has meaning. Use **one-hot encoding** instead.

### Hyperparameter tuning
- Grid search