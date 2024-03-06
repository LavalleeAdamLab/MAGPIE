#!/usr/bin/env python3
# Emily Hashimoto-Roth ☆
# Lavallée-Adam Lab
# Logistic regression machine learning model
# Python 3.7


import Utilities as af
from sklearn.linear_model import LogisticRegression


def run_model(positive_training_examples_path,
              negative_training_examples_path,
              positive_testing_examples_path,
              negative_testing_examples_path):

    """ FUNCTION DESCRIPTION ❀
    Trains and tests a logistic regression model for assessing the confidence of a
    putative protein-protein interaction (i.e., determining the probability that a given
    interaction is bona fide or a product of non-specific binding).

    :param positive_training_examples_path: path to positive training examples
    :type positive_training_examples_path: directory path / str
    :param negative_training_examples_path: path to negative training examples
    :type negative_training_examples_path: directory path / str
    :param positive_testing_examples_path: path to positive testing examples
    :type positive_testing_examples_path: directory path / str
    :param negative_testing_examples_path: path to negative testing (or validation) examples
    :type negative_testing_examples_path: directory path / str
    :return: nested list, nested list
    """

    positive_training_examples = af.open_examples(positive_training_examples_path)
    negative_training_examples = af.open_examples(negative_training_examples_path)
    positive_testing_examples = af.open_examples(positive_testing_examples_path)
    negative_testing_examples = af.open_examples(negative_testing_examples_path)

    # Concatenate training examples, subset features and labels
    all_training_examples = positive_training_examples + negative_training_examples
    x_train = [all_training_examples[i][1:-1] for i in range(len(all_training_examples))]  # Features
    y_train = [all_training_examples[i][-1] for i in range(len(all_training_examples))]  # Labels

    # Create classifier object and train
    # Setting multi_class argument to 'multinomial' uses cross-entropy loss
    # Sci-kit learn documentation states 'multinomial' can be used for binary classification
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    classifier = LogisticRegression(penalty='l2', multi_class='multinomial',
                                    solver='newton-cg').fit(x_train, y_train)
    model_weights = af.get_weights(classifier.coef_)

    # Predict probabilities for test examples
    # Because 'multinomial' is specified, the softmax function is used to find the predicted probability of each class
    positive_test = [positive_testing_examples[i][1:-1] for i in range(len(positive_testing_examples))]
    model_positive_predictions_arr = classifier.predict_proba(positive_test)
    model_positive_predictions = model_positive_predictions_arr.tolist()
    for i in range(len(model_positive_predictions)):
        model_positive_predictions[i].insert(0, positive_testing_examples[i][0])
    model_positive_predictions.insert(0, ['Input', 'Probability_Interaction', 'Probability_Non_Specific_Binding'])

    negative_test = [negative_testing_examples[i][1:-1] for i in range(len(negative_testing_examples))]
    model_negative_predictions_arr = classifier.predict_proba(negative_test)
    model_negative_predictions = model_negative_predictions_arr.tolist()
    for i in range(len(model_negative_predictions)):
        model_negative_predictions[i].insert(0, negative_testing_examples[i][0])
    model_negative_predictions.insert(0, ['Input', 'Probability_Interaction', 'Probability_Non_Specific_Binding'])

    return model_positive_predictions, model_negative_predictions, model_weights
