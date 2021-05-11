import pandas as pd
from numpy import nan
from sklearn.preprocessing import MinMaxScaler
from string import ascii_uppercase

def basic():
    dataset = pd.read_csv('titanic.csv')

    # Drop those for the basic preprocessing
    dataset = dataset.drop(labels=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 'columns')

    # Drop rows - only 2 missing
    dataset = dataset[dataset['Embarked'].notna()]
    # Fill Age with mean
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
    dataset['Age'] = dataset['Age'].astype('int')

    # Cast Fare to int
    dataset['Fare'] = dataset['Fare'].astype('int')
    # Change cathegorical features to dummy values
    dataset = dataset.join(pd.get_dummies(dataset['Sex'], prefix='Sex').astype('int'))
    dataset = dataset.drop(['Sex'], axis = 'columns')

    dataset = dataset.join(pd.get_dummies(dataset['Embarked'], prefix='Embarked').astype('int'))
    dataset = dataset.drop(['Embarked'], axis = 'columns')

    # Scale values
    scaler = MinMaxScaler()
    dataset[['Age', 'Fare']] = scaler.fit_transform(dataset[['Age', 'Fare']])

    return dataset['Survived'], dataset.drop(['Survived'], axis='columns')


def age_divider(age):
    if age < 10:
        return "child"
    if age < 20:
        return "teenager"
    if age < 30:
        return "young adult"
    if age < 60:
        return "adult"
    return "eldery"

def title(name):
    titles = ['Mr', 'Mrs', 'Miss', 'Master']
    for t in titles:
        if t in name:
            return t
    return 'none'

def cabin_letter(cabin):
    if cabin == 'none':
        return 'none'
    for anum in cabin:
        if anum in ascii_uppercase:
            return anum

def advanced():
    dataset = pd.read_csv('titanic.csv')

    dataset = dataset.drop(labels=['PassengerId'], axis = 'columns')

    # Drop rows - only 2 missing
    dataset = dataset[dataset['Embarked'].notna()]
    dataset.drop('Ticket', axis='columns', inplace=True)
    # Fill Age with mean
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
    dataset['Age'] = dataset['Age'].astype('int')
    
    # Add age group
    dataset['Age'] = [age_divider(x) for x in dataset['Age']]

    # Add tile 
    dataset['Title'] = [title(x) for x in dataset['Name']]
    dataset.drop('Name', axis='columns', inplace=True)

    # Add cabin
    # dataset.drop('Cabin', axis='columns', inplace=True)
    dataset['Cabin'] = dataset['Cabin'].fillna('none')
    dataset['Cabin'] = [cabin_letter(x) for x in dataset['Cabin'].astype('string')]

    # Change cathegorical features to dummy values
    dataset = dataset.join(pd.get_dummies(dataset['Cabin'], prefix='Cabin').astype('int'))
    dataset = dataset.drop(['Cabin'], axis = 'columns')

    dataset = dataset.join(pd.get_dummies(dataset['Title'], prefix='Title').astype('int'))
    dataset = dataset.drop(['Title'], axis = 'columns')

    dataset = dataset.join(pd.get_dummies(dataset['Age'], prefix='Age').astype('int'))
    dataset = dataset.drop(['Age'], axis = 'columns')

    dataset = dataset.join(pd.get_dummies(dataset['Sex'], prefix='Sex').astype('int'))
    dataset = dataset.drop(['Sex'], axis = 'columns')

    dataset = dataset.join(pd.get_dummies(dataset['Embarked'], prefix='Embarked').astype('int'))
    dataset = dataset.drop(['Embarked'], axis = 'columns')

    # Scale values
    scaler = MinMaxScaler()
    dataset[['Fare']] = scaler.fit_transform(dataset[['Fare']])


    return dataset['Survived'], dataset.drop(['Survived'], axis='columns')



def basic_plus():
    dataset = pd.read_csv('titanic.csv')

    # Drop those for the basic preprocessing
    dataset = dataset.drop(labels=['PassengerId', 'Ticket', 'Cabin'], axis = 'columns')

    # Drop rows - only 2 missing
    dataset = dataset[dataset['Embarked'].notna()]
    # Fill Age with mean
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
    dataset['Age'] = dataset['Age'].astype('int')

    # Cast Fare to int
    dataset['Fare'] = dataset['Fare'].astype('int')

    # Add tile 
    dataset['Title'] = [title(x) for x in dataset['Name']]
    dataset.drop('Name', axis='columns', inplace=True)

    # Change cathegorical features to dummy values
    dataset = dataset.join(pd.get_dummies(dataset['Sex'], prefix='Sex').astype('int'))
    dataset = dataset.drop(['Sex'], axis = 'columns')

    dataset = dataset.join(pd.get_dummies(dataset['Title'], prefix='Title').astype('int'))
    dataset = dataset.drop(['Title'], axis = 'columns')

    dataset = dataset.join(pd.get_dummies(dataset['Embarked'], prefix='Embarked').astype('int'))
    dataset = dataset.drop(['Embarked'], axis = 'columns')

    # Scale values
    scaler = MinMaxScaler()
    dataset[['Age', 'Fare']] = scaler.fit_transform(dataset[['Age', 'Fare']])

    return dataset['Survived'], dataset.drop(['Survived'], axis='columns')

def corelations():
    dataset = pd.read_csv('titanic.csv')
    # Check corelations
    print(dataset[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
    print(dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
    print(dataset[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())
    print(dataset[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())