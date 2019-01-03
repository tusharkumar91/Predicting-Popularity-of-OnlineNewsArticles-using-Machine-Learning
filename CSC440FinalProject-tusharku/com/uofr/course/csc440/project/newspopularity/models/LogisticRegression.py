import torch
import torch.nn as nn
import torch.autograd.variable as Variable
import numpy as np


class LogisticRegressionModel(nn.Module):
    r"""
    This class encapsulates all the mechanics of
    running Logistic Regression on the online news article
    dataset.
    """
    def __init__(self, input_dim, output_dim, train_data, test_data,
                 learning_rate, epochs, batch_size, weight_decay,
                 log_interval):
        r"""
        Initiale the model with all the parameters required for training and
        testing the logistic regression model
        :param input_dim: dimensions of input data, same as number of attributes in your data
        :param output_dim: dimensions of output data or classes you want to predict
        :param train_data: Training data
        :param test_data: Dataset to be used for testing
        :param learning_rate: leraning rate to be used while learning
        :param epochs: Number of epochs to run the training for
        :param batch_size: Batch size to be used while learning
        :param log_interval: After how many epochs must the accuracy be reported.
        """
        super(LogisticRegressionModel, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.weight_decay = weight_decay
        self.best_weight = None
        self.best_accuracy = 0.0

    def forward(self, x):
        return self.linear(x)

    def train_model(self):
        r"""
        Method to train the logistic regression model
        :return: None
        """
        crieterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        params = list(self.parameters())
        for i in range(len(params)):
            print(params[i].size())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, patience=50)
        for epoch in range(1, self.epochs+1):
            X, Y = self.train_data
            iter = 1
            for batch_start in np.arange(0, len(X), self.batch_size):
                x = X[batch_start: min(batch_start+self.batch_size, len(X))]
                y = Y[batch_start: min(batch_start+self.batch_size, len(X))]
                optimizer.zero_grad()
                inputs = Variable(torch.tensor(x).float().view(-1, self.input_dim))
                labels = Variable(torch.LongTensor(y))
                outputs = self.forward(inputs)
                loss = crieterion(outputs, labels)
                loss.backward()
                optimizer.step()
            scheduler.step(loss.data)
            print("Epoch : {}, Training Loss : {}".format(epoch, loss.data))
            if epoch % self.log_interval == 0:
                accuracy = self.test_model()
                if accuracy > self.best_accuracy:
                    self.best_weight = self.state_dict()
                    self.best_accuracy = accuracy

    def test_model(self):
        r"""
        Method to test the learnt model using the logistic regression
        learning algorithm on unseen data
        :return:
        """
        correct_count = 0.0
        total_count = 0.0
        X, Y = self.test_data
        for idx in range(len(X)):
            x = X[idx]
            y = Y[idx]
            inputs = torch.tensor(x).float().view(1, -1)
            outputs = self.forward(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total_count += 1
            correct_count += ((predicted.numpy()[0] == y).sum())

        accuracy = (100.0 * correct_count) / total_count
        print("Test Accuracy : {}".format(accuracy))
        return accuracy

    def get_performance_metrics(self):
        r"""
        Method to get the best weights and accuracy during the learning
        process. Typically this should ONLY be extracted when test data
        is actually validation data
        :return:
        """
        return self.best_accuracy, self.best_weight
