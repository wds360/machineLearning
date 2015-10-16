import csv

label_idx = 'Survived'
embarked_labels = {'':'0', 'S':'0','C':'1', 'Q':'2'}
selected_feature_idx = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

labels = []
features = []

with open("titanic.csv") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        labels.append(row[label_idx])

        pclass = row['Pclass']
        sex = '0' if row['Sex'] is 'male' else '1'
        age = '-1' if row['Age'] is '' else row['Age']
        sibsp = row['SibSp']
        parch = row['Parch']
        fare = row['Fare']
        embarked = embarked_labels[row['Embarked']]

        features.append([pclass,sex,age,sibsp,parch,fare,embarked])
