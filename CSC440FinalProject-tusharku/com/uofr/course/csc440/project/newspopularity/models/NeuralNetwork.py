import torch
import torch.nn as nn
import torch.autograd.variable as Variable
import numpy as np


class NeuralNetworkModel(nn.Module):
    r"""
    Class to encapsulate the mechanics of running neural
    network algorithm on the online news dataset
    """
    def __init__(self, input_dim, hidden_layers, hidden_dims, non_linearity, output_dim,
                 epochs, learning_rate, batch_size, dropout_p, weight_decay,
                 log_interval, train_data, test_data):
        r"""
        Method to initializa the necessary parameters for training the neural network
        :param input_dim: dimensions of input data, same as number of attributes in your data
        :param output_dim: dimensions of output data or classes you want to predict
        :param train_data: Training data
        :param test_data: Dataset to be used for testing
        :param learning_rate: leraning rate to be used while learning
        :param epochs: Number of epochs to run the training for
        :param batch_size: Batch size to be used while learning
        :param log_interval: After how many epochs must the accuracy be reported.
        :param hidden_layers: Number of hidden layers to be used
        :param hidden_dims: Dimensions of the hidden layers in the order of layers
        :param non_linearity: Non linearity to be used for activations
        :param dropout_p: dropout probability to ensure generalization of model
        :param weight_decay: Weight decay for regularization purposes
        """
        super(NeuralNetworkModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_dims = hidden_dims
        self.train_data = train_data
        self.test_data = test_data
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = torch.nn.Dropout(dropout_p)
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        input_size = input_dim
        self.layer_modules = torch.nn.ModuleList()
        for layer in range(hidden_layers):
            linear_layer = torch.nn.Linear(input_size, hidden_dims[layer])
            torch.nn.init.xavier_uniform_(linear_layer.weight)
            torch.nn.init.zeros_(linear_layer.bias)
            self.layer_modules.append(torch.nn.Sequential(linear_layer,
                                                          non_linearity))
            input_size = hidden_dims[layer]
        final_linear_layer = torch.nn.Linear(input_size, output_dim)
        torch.nn.init.xavier_uniform_(final_linear_layer.weight)
        torch.nn.init.zeros_(final_linear_layer.bias)
        self.layer_modules.append(final_linear_layer)
        self.best_weight = None
        self.best_accuracy = 0.0

    def forward(self, x):
        out = None
        for module in self.layer_modules[0:len(self.layer_modules)-1]:
            out = module(x)
            out = self.dropout(out)
            x = out
        out = self.layer_modules[len(self.layer_modules)-1](x)
        return out

    def train_model(self):
        r"""
        Method to train the neural network with the parameters
        that it has been initialized
        :return:
        """
        self.train()
        crieterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               verbose=True, patience=10, factor=0.2,
                                                               threshold=1E-4, eps=1E-10)
        for epoch in range(self.epochs):
            X, Y = self.train_data
            for batch_start in np.arange(0, len(X), self.batch_size):
                x = X[batch_start: min(batch_start+self.batch_size, len(X))]
                y = Y[batch_start: min(batch_start+self.batch_size, len(X))]
                optimizer.zero_grad()
                inputs = Variable(torch.tensor(x).float().view(-1, self.input_dim))
                labels = Variable(torch.LongTensor(y))
                outputs = self.forward(inputs)
                loss = crieterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
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
        Method to test the neural network using the weights it has learned
        for the prediction task
        :return:
        """
        self.eval()
        correct_count = 0.0
        total_count = 0.0
        X, Y = self.test_data
        for idx in range(len(X)):
            x = X[idx]
            y = Y[idx]
            inputs = Variable(torch.tensor(x).float().view(1, -1))
            outputs = self.forward(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total_count += 1
            correct_count += ((predicted.numpy()[0] == y).sum())

        accuracy = (100.0 * correct_count) / total_count
        print("Accuracy : {}".format(accuracy))
        self.train()
        return accuracy

    def get_performance_metrics(self):
        r"""
        Method to get the best weights and accuracy during the learning
        process. Typically this should ONLY be extracted when test data
        is actually validation data
        :return:
        """
        return self.best_accuracy, self.best_weight
