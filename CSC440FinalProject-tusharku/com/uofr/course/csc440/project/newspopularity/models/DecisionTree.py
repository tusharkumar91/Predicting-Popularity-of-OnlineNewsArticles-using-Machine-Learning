from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score


class DecisionTreeModel:
    r"""
    Class to encapsulate the mechanics of running extra
    tree classification algorithm on the online news dataset
    """
    def __init__(self, train_data, test_data, max_depth):
        self.classifier = DecisionTreeClassifier(max_depth=max_depth)
        self.x_train, self.y_train = train_data
        self.x_test, self.y_test = test_data

    def train_model(self):
        self.classifier.fit(self.x_train, self.y_train)

    def test_model(self):
        y_pred = self.classifier.predict(self.x_test)
        print("="*40)
        print("Classification Report")
        print("="*40)
        print(classification_report(self.y_test, y_pred))
        accuracy = 100 * accuracy_score(self.y_test, y_pred)
        return accuracy



