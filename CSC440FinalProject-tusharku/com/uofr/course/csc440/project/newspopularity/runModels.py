

from com.uofr.course.csc440.project.newspopularity.dataloader.NewsDataLoader import NewsDataLoader
from com.uofr.course.csc440.project.newspopularity.models.LogisticRegression import LogisticRegressionModel
from com.uofr.course.csc440.project.newspopularity.models.NeuralNetwork import NeuralNetworkModel
from com.uofr.course.csc440.project.newspopularity.models.RandomForestClassifier import RandomForestClassifierModel
from com.uofr.course.csc440.project.newspopularity.models.DecisionTree import DecisionTreeModel
from com.uofr.course.csc440.project.newspopularity.models.ExtraTreeClassifier import ExtraTreesClassifierModel

import time
import argparse
import torch
import numpy as np
import os.path


NUM_CLASSES = 2
DATA_DIR = "data"
TARGET_LABEL = "shares"
URL_ATTRIBUTE_NAME = "url"
GLOVE_FILE_PATH = "com/uofr/course/csc440/project/newspopularity/data/glove.twitter.27B.25d.txt"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CSC440 Project')

    # Program related settings

    parser.add_argument('--train_test_split', default=0.2, type=float,
                        help='Split to be used nnet training and testing')
    parser.add_argument('--model', default="dtree", type=str,
                        help='Model to use for learning')

    # Training procedure settings

    parser.add_argument('--log_interval', type=int, default=10,
                        help='report interval after N epochs')
    parser.add_argument('--k', type=int, default=61,
                        help='Top K ranked features to select for learning')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=150,
                        help='upper epoch limit')
    parser.add_argument('--lr', '--learning-rate', default=1E-3, type=float,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout probability for Neural Network')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight Decay to be used ')
    parser.add_argument('--max_depth_tree', type=int, default=5,
                        help='depth till which tree should be grown for decision trees')
    parser.add_argument('--max_depth_forest', type=int, default=26,
                        help='depth till which tree should be grown for random forest')
    parser.add_argument('--max_depth_extra_trees', type=int, default=22,
                        help='depth till which tree should be grown for extra trees')
    parser.add_argument('--trees_forest', default=550, type=int,
                        help='Number of trees to learn for random forest')
    parser.add_argument('--trees_extra', default=500, type=int,
                        help='Number of trees to learn for extra trees classifier')
    parser.add_argument('--use_embedding', default=False, action="store_true",
                        help='Should glove word embeddings be used')

    args = parser.parse_args()

    args_dict = vars(args)

    print('\n\nArgument list to program\n\n')

    print('\n'.join(['--{0} {1}'.format(arg, args_dict[arg])
                     for arg in args_dict]))

    # Set the seed value

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #############
    # LOAD DATA #
    #############

    dataset_path = "com/uofr/course/csc440/project/newspopularity/data/OnlineNewsPopularity.csv"
    data = NewsDataLoader.fetch_data(file_path=dataset_path, sep=",", header=0, na_filter=False)
    NewsDataLoader.format_columns(data_frame=data)
    NewsDataLoader.bin_target_label(target_key=TARGET_LABEL, bin_classes=NUM_CLASSES, data_frame=data)
    data = NewsDataLoader.get_data_with_top_k_ranked_features(data_frame=data, k=args.k)

    #######################
    # SPLIT TRAINING/TEST #
    #######################

    train_data, _, test_data = NewsDataLoader.train_validate_test_split(df=data,
                                                                        train_percent=(1-args.train_test_split),
                                                                        validate_percent=0,
                                                                        seed=args.seed)

    if args.use_embedding:
        word_vector = {}
        print("\nLoading Glove Vectors\n")
        if not os.path.exists(GLOVE_FILE_PATH):
            raise ValueError("Please download the glove vector file as its not present at " + GLOVE_FILE_PATH)
        f = open(GLOVE_FILE_PATH, 'r', encoding="utf8")
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            word_vector[word] = embedding
        print("\nDone Loading Glove Vectors\n")

    #######################
    # SCALE TRAINING DATA #
    #######################

    use_embeddings = False
    if args.use_embedding:
        if URL_ATTRIBUTE_NAME in train_data.keys():
            word_vector_df = np.ndarray(shape=(train_data.values.shape[0], 25))
            index = 0
            for _, row in train_data.iterrows():
                url_vector = [0] * 25
                url = row['url']
                words = url.rsplit("/", 2)
                words = words[1].split("-")
                for word in words:
                    if word in word_vector:
                        url_vector = np.add(url_vector, word_vector[word])
                word_vector_df[index] = url_vector
                index += 1
            train_data = train_data.drop(URL_ATTRIBUTE_NAME, axis=1)
            for i in range(25):
                train_data["url_vec"+str(i)] = word_vector_df[:, i]
            use_embeddings = True
    else:
        if URL_ATTRIBUTE_NAME in train_data.keys():
            train_data = train_data.drop(URL_ATTRIBUTE_NAME, axis=1)
    train_scaled = (train_data - train_data.mean()) / (train_data.max() - train_data.min())
    train_scaled[TARGET_LABEL] = train_data[TARGET_LABEL]
    if use_embeddings:
        for i in range(25):
            train_scaled["url_vec" + str(i)] = train_data["url_vec" + str(i)]

    train_data_x = train_scaled.drop([TARGET_LABEL], axis=1).values
    train_data_y = train_scaled.loc[:, TARGET_LABEL].values

    #######################
    # SCALE TESTING DATA #
    #######################

    if args.use_embedding:
        if URL_ATTRIBUTE_NAME in test_data.keys():
            word_vector_df = np.ndarray(shape=(test_data.values.shape[0], 25))
            index = 0
            for _, row in test_data.iterrows():
                url_vector = [0] * 25
                url = row['url']
                words = url.rsplit("/", 2)
                words = words[1].split("-")
                for word in words:
                    if word in word_vector:
                        url_vector = np.add(url_vector, word_vector[word])
                word_vector_df[index] = url_vector
                index += 1
            test_data = test_data.drop(URL_ATTRIBUTE_NAME, axis=1)
            for i in range(25):
                test_data["url_vec"+str(i)] = word_vector_df[:, i]
    else:
        if URL_ATTRIBUTE_NAME in test_data.keys():
            test_data = test_data.drop(URL_ATTRIBUTE_NAME, axis=1)
    test_scaled = (test_data - test_data.mean()) / (test_data.max() - test_data.min())
    test_scaled[TARGET_LABEL] = test_data[TARGET_LABEL]
    if use_embeddings:
        for i in range(25):
            test_scaled["url_vec" + str(i)] = test_scaled["url_vec" + str(i)]
    test_data_x = test_scaled.drop([TARGET_LABEL], axis=1).values
    test_data_y = test_scaled.loc[:, TARGET_LABEL].values

    NUM_ATTRIBUTES = len(test_scaled.keys())-1
    best_acc = 0.0
    best_depth = 0
    start_time = time.time()
    print("\n")
    if args.model == "dtree":
        model = DecisionTreeModel(train_data=(train_data_x, train_data_y), test_data=(test_data_x, test_data_y),
                                  max_depth=args.max_depth_tree)
    elif args.model == "logistic":
        model = LogisticRegressionModel(train_data=(train_data_x, train_data_y),
                                        test_data=(test_data_x, test_data_y),
                                        learning_rate=args.lr, input_dim=NUM_ATTRIBUTES,
                                        output_dim=NUM_CLASSES, epochs=args.epochs, batch_size=args.batch_size,
                                        weight_decay=args.weight_decay, log_interval=args.log_interval)
    elif args.model == "nnet":
        model = NeuralNetworkModel(input_dim=NUM_ATTRIBUTES, output_dim=NUM_CLASSES,
                                   train_data=(train_data_x, train_data_y),test_data=(test_data_x, test_data_y),
                                   learning_rate=args.lr,epochs=args.epochs, batch_size=args.batch_size,
                                   hidden_dims=[100],hidden_layers=1, dropout_p=args.dropout,
                                   weight_decay=args.weight_decay, non_linearity=torch.nn.LeakyReLU(),
                                   log_interval=args.log_interval)
    elif args.model == "etree":
        model = ExtraTreesClassifierModel(train_data=(train_data_x, train_data_y),test_data=(test_data_x, test_data_y),
                                          n_estimators=args.trees_extra,
                                          max_depth=args.max_depth_extra_trees)
    elif args.model == "forest":
        model = RandomForestClassifierModel(train_data=(train_data_x, train_data_y),test_data=(test_data_x, test_data_y),
                                            n_estimators=args.trees_forest,
                                            max_depth=args.max_depth_forest)
    else:
        raise ValueError("model can only be dtree/logistic/nnet/etree/forest but was provided " + args.dataset)

    model.train_model()
    print("\nTime taken to train the model (seconds) : {}\n".format(time.time() - start_time))
    accuracy = model.test_model()
    print("\nAccuracy achieved by the model on test set : {}\n".format(accuracy))



