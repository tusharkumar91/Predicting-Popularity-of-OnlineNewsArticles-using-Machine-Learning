from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class ExtraTreesClassifierModel:
    r"""
    Class to encapsulate the mechanics of running extra
    tree classification algorithm on the online news dataset
    """
    def __init__(self, train_data,
                 test_data,
                 n_estimators,
                 max_depth):
        self.classifier = ExtraTreesClassifier(n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               random_state=0, n_jobs=10,
                                                 )
        self.x_train, self.y_train = train_data
        self.x_test, self.y_test = test_data

    def train_model(self):
        print("Training the Extra Trees Classifier Model")
        self.classifier.fit(self.x_train, self.y_train)

    def test_model(self):
        y_pred = self.classifier.predict(self.x_test)
        print("=" * 40)
        print("Classification Report")
        print("=" * 40)
        print(classification_report(self.y_test, y_pred))
        accuracy = 100 * accuracy_score(self.y_test, y_pred)
        return accuracy

    def predict(self, x):
        return self.classifier.predict(x)



