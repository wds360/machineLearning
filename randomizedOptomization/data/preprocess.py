import csv
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

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

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

X_train = [[float(x) for x in row] for row in X_train]
X_test = [[float(x) for x in row] for row in X_test]

# fit a standardScaler to normalize all input to zero mean and unit variance
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

def write_array(filename, array):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in array:
            writer.writerow(row)

write_array('titanicTrain.csv',X_train)
write_array('titanicTrainLabels.csv',y_train)
write_array('titanicTest.csv',X_test)
write_array('titanicTestLabels.csv',y_test)
