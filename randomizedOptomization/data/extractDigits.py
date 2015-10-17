import csv
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer

# The digits dataset
digits = datasets.load_digits()

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
labels = LabelBinarizer().fit_transform(digits.target)


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)

def write_array(filename, array):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in array:
            writer.writerow(row)

write_array('digitsTrain.csv',X_train)
write_array('digitsTrainLabels.csv',y_train)
write_array('digitsTest.csv',X_test)
write_array('digitsTestLabels.csv',y_test)

