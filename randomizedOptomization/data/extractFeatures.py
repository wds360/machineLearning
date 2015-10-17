import csv
from sklearn.cross_validation import train_test_split

labels = []
features = []

with open("titanic2.csv") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        labels.append(row['survived'])

        firstclass = row['firstclass']
        secondclass = row['secondclass']
        thirdclass = row['thirdclass']
        crew = row['crew']
        child = row['child']
        female = row['female']

        features.append([firstclass,secondclass,thirdclass,crew,child,female])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

def write_array(filename, array):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in array:
            writer.writerow(row)

write_array('titanicTrain.csv',X_train)
write_array('titanicTrainLabels.csv',y_train)
write_array('titanicTest.csv',X_test)
write_array('titanicTestLabels.csv',y_test)
