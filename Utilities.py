#!/usr/bin/env python3
# Emily Hashimoto-Roth ☆
# Lavallée-Adam Lab
# Model functions for analysis
# Python 3.7


import csv
import random


def transpose_dataset(dataset):

    """ FUNCTION DESCRIPTION ❀
    Transposes a nested list containing the spectral count data for various experiments, whose columns
    represent experiment runs and rows represent protein identifications. The reverse is true upon
    return.

    :param dataset: spectral count data
    :type dataset: nested list
    :return: nested list
    """

    transposed_dataset = [[dataset[j][i] for j in range(len(dataset))] for i in range(len(dataset[0]))]

    return transposed_dataset


def add_pseudo_count(dataset_row):

    """ FUNCTION DESCRIPTION ❀
    Adds a pseudo-count to the spectral count data of a control run, such that zero values are replaced
    by randomly sampling a value from the bottom 10% of non-zero values.

    :param dataset_row: spectral count data of a control run
    :type dataset_row: list
    :return: list
    """

    random.seed(0)

    tmp = []
    for i in range(1, len(dataset_row)):
        if dataset_row[i] != 0:
            tmp.append(dataset_row[i])
    tmp_sorted = sorted(tmp)

    bottom_index = int((len(tmp_sorted) / 10))
    bottom_nonzero_values = tmp_sorted[0:bottom_index]

    for i in range(1, len(dataset_row)):
        if dataset_row[i] == 0:
            dataset_row[i] = random.choice(bottom_nonzero_values)
            # dataset_row[i] = float(1.0)

    return dataset_row


def calculate_zscore(data, mean, standard_deviation):

    """ FUNCTION DESCRIPTION ❀
    Computes a Z-score using the inputted data. If the standard deviation is zero, the Z-score
    will be zero, avoiding a division by zero error.

    :param data: spectral count value
    :type data: float
    :param mean: spectral count mean
    :type mean: float
    :param standard_deviation: spectral count standard deviation
    :type standard_deviation: float
    :return: float
    """

    data = float(data)
    mean = float(mean)
    standard_deviation = float(standard_deviation)

    if standard_deviation == 0:
        zscore = 0
    else:
        zscore = round((data - mean) / standard_deviation, 10)

    return zscore


def calculate_foldchange(data, mean):

    """ FUNCTION DESCRIPTION ❀
    Computes a fold-change value using the inputted data. If the mean is zero, the fold-change value
    will be zero, avoiding a division by zero error.

    :param data: spectral count value
    :type data: float
    :param mean: spectral count mean
    :type mean: float
    :return: float
    """

    data = float(data)
    mean = float(mean)

    if mean == 0:
        foldchange = 0
    elif data == 0:
        foldchange = 0
    else:
        foldchange = round(data / mean, 10)

    return foldchange


def fdr_monotonic_transformation(rates):

    """ FUNCTION DESCRIPTION ❀
    Applies a monotonic transformation to the inputted data, such that the lowest false discovery
    rate is recorded as the function iterates through the list.

    :param rates: estimated false discovery rates
    :type rates: list
    :return: list
    """

    for i in range(1, len(rates)):
        if rates[i] > rates[i - 1]:
            rates[i] = rates[i - 1]
        else:
            continue

    return list(rates)


def confidence_analysis_formatting(thresholds,
                                   experimental_count,
                                   control_count,
                                   experimental_count_normalized,
                                   control_count_normalized,
                                   fdr,
                                   fdr_smooth):

    """ FUNCTION DESCRIPTION ❀
    Formats the results of a confidence score analysis (Z-score or fold-change analysis) to save them.

    :param thresholds: thresholds wherein confidence scores were computed
    :type thresholds: list
    :param experimental_count: number of detections in the experiments at or above a given threshold
    :type experimental_count: list
    :param control_count: number of detections in the control at or above a given threshold
    :type control_count: list
    :param experimental_count_normalized: normalized number of detections in the experiments at or
    above a given threshold
    :type experimental_count_normalized: list
    :param control_count_normalized: normalized number of detections in the control at or above a given
    threshold
    :type control_count_normalized: list
    :param fdr: computed false discovery rates for each given threshold
    :type fdr: list
    :param fdr_smooth: computed false discovery rates for each given threshold with monotonic transformation
    :type fdr_smooth: list
    :return: zip
    """

    Col1 = thresholds
    Col1.insert(0, 'Thresholds')
    Col2 = experimental_count
    Col2.insert(0, 'ExperimentCount')
    Col3 = control_count
    Col3.insert(0, 'ControlCount')
    Col4 = experimental_count_normalized
    Col4.insert(0, 'ExperimentCountNormalized')
    Col5 = control_count_normalized
    Col5.insert(0, 'ControlCountNormalized')
    Col6 = fdr
    Col6.insert(0, 'FDR')
    Col7 = fdr_smooth
    Col7.insert(0, 'FDRsmooth')
    colList = zip(Col1, Col2, Col3, Col4, Col5, Col6, Col7)

    return colList


def loocv_analysis_formatting(thresholds,
                              experimental_count_normalized,
                              control_count_normalized,
                              fdr,
                              fdr_smooth):

    """ FUNCTION DESCRIPTION ❀
    Formats the results of a confidence score analysis (Z-score or fold-change analysis) to save them.

    :param thresholds: thresholds wherein confidence scores were computed
    :type thresholds: list
    :param experimental_count_normalized: normalized number of detections in the experiments at or
    above a given threshold
    :type experimental_count_normalized: list
    :param control_count_normalized: normalized number of detections in the control at or above a given
    threshold
    :type control_count_normalized: list
    :param fdr: computed false discovery rates for each given threshold
    :type fdr: list
    :param fdr_smooth: computed false discovery rates for each given threshold with monotonic transformation
    :type fdr_smooth: list
    :return: zip
    """

    Col1 = thresholds
    Col1.insert(0, 'Thresholds')
    Col2 = experimental_count_normalized
    Col2.insert(0, 'NormalizedPositiveSum')
    Col3 = control_count_normalized
    Col3.insert(0, 'NormalizedNegativeSum')
    Col4 = fdr
    Col4.insert(0, 'FDR')
    Col5 = fdr_smooth
    Col5.insert(0, 'FDRsmooth')
    colList = zip(Col1, Col2, Col3, Col4, Col5)

    return colList


def model_building_formatting_list(build_objects_list):

    """ FUNCTION DESCRIPTION ❀
    Formats the results of dataset building (training, testing) to save them.

    :param build_objects_list: Build objects corresponding a given run
    :type build_objects_list: list
    :return: zip
    """

    examples_bait_prey = []
    examples_feature1 = []
    examples_feature2 = []
    examples_feature3 = []
    examples_feature4 = []
    examples_feature5 = []
    examples_labels = []
    for i in range(len(build_objects_list)):
        for j in range(len(build_objects_list[i].bait_preys)):
            examples_bait_prey.append(build_objects_list[i].bait_preys[j])
            examples_feature1.append(build_objects_list[i].feature1[j])
            examples_feature2.append(build_objects_list[i].feature2[j])
            examples_feature3.append(build_objects_list[i].feature3[j])
            examples_feature4.append(build_objects_list[i].feature4[j])
            examples_feature5.append(build_objects_list[i].feature5[j])
            examples_labels.append(build_objects_list[i].labels[j])
    Col1 = examples_bait_prey.copy()
    Col1.insert(0, 'BaitPrey')
    Col2 = examples_feature1.copy()
    Col2.insert(0, 'StandardizedCount')
    Col3 = examples_feature2.copy()
    Col3.insert(0, 'ControlAverage')
    Col4 = examples_feature3.copy()
    Col4.insert(0, 'ControlStandardDeviation')
    Col5 = examples_feature4.copy()
    Col5.insert(0, 'ControlMax')
    Col6 = examples_feature5.copy()
    Col6.insert(0, 'FoldChange')
    Col7 = examples_labels.copy()
    Col7.insert(0, 'Class')
    colList = zip(Col1, Col2, Col3, Col4, Col5, Col6, Col7)

    return colList


def false_positive_building(build_object):

    """ FUNCTION DESCRIPTION ❀
    Formats the results of dataset building (validation) to save them.

    :param build_object: object created for the control left out during a leave-one-out run
    :type build_object: Build
    :return: zip
    """

    Col1 = build_object.bait_preys.copy()
    Col1.insert(0, 'BaitPrey')
    Col2 = build_object.feature1.copy()
    Col2.insert(0, 'StandardizedCount')
    Col3 = build_object.feature2.copy()
    Col3.insert(0, 'ControlAverage')
    Col4 = build_object.feature3.copy()
    Col4.insert(0, 'ControlStandardDeviation')
    Col5 = build_object.feature4.copy()
    Col5.insert(0, 'ControlMax')
    Col6 = build_object.feature5.copy()
    Col6.insert(0, 'FoldChange')
    Col7 = build_object.labels.copy()
    Col7.insert(0, 'Class')
    colList = zip(Col1, Col2, Col3, Col4, Col5, Col6, Col7)

    return colList


def open_examples(examples_file_path):

    """ FUNCTION DESCRIPTION ❀
    Opens and formats the examples appropriately for use with machine learning model.

    :param examples_file_path: path to examples (training, validation, or testing) and their features
    and labels
    :type examples_file_path: directory path / str
    :return: nested list
    """

    examples_file = []
    with open(examples_file_path, 'r') as f1:
        content = csv.reader(f1, delimiter=',')
        for i in content:
            examples_file.append(i)
    examples = examples_file[1:]  # Omit header
    # Convert strings to floats where appropriate
    for i in range(len(examples)):
        for j in range(1, len(examples_file[i]) - 1):
            examples[i][j] = float(examples[i][j])

    return examples


def get_weights(model_coefficients):

    """ FUNCTION DESCRIPTION ❀
    Converts and formats array of coefficents outputted by SciKit-Learn's logistic regression model,
    after it's been trained.

    :param model_coefficients: SciKit-Learn attribute for trained logistic regression model
    :type model_coefficients: numpy array
    :return: list
    """

    model_coefficients_list = model_coefficients.tolist()
    model_coefficients_list_flat = [round(item, 10) for sublist in model_coefficients_list for item in sublist]

    return model_coefficients_list_flat


def format_weights(model_coefficients_list):

    """ FUNCTION DESCRIPTION ❀

    Formats features and feature weights from logistic regression model.

    :param model_coefficients_list: feature (model) coefficient from logistic regression model
    :type model_coefficients_list: list
    :return: nested list
    """

    features = ['SpectralCount',
                'AvgSpectralCount',
                'StnDevSpectralCount',
                'MaxSpectralCount',
                'FoldChange']

    features_weights = []
    for i in range(len(model_coefficients_list)):
        features_weights.append([features[i], str(model_coefficients_list[i])])

    return features_weights
