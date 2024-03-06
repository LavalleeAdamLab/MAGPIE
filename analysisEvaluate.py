#!/usr/bin/env python3
# Emily Hashimoto-Roth ☆
# Lavallée-Adam Lab
# Model analysis for evaluating calculated values (confidence scores or probabilities)
# Python 3.7


import copy
import Utilities as af


def evaluate(experiment_data, control_data, thresholds):

    """ FUNCTION DESCRIPTION ❀
    Given a list of confidence scores or probabilities for putative interactions and detections in controls
    and a list of thresholds, false discovery rates (FDRs) are estimated at each threshold. The number of
    confidence scores or probabilities belonging to putative interactions and detections in controls are
    determined and normalized at each threshold in order to estimate FDRs.

    :param experiment_data: values (confidence scores or probabilities) calculated for putative interactions
    :type experiment_data: list
    :param control_data: values (confidence scores or probabilities) calculated for detections in the controls
    :type control_data: list
    :param thresholds: range and resolution at which to evaluate values (confidence scores or probabilities)
    :type thresholds: list
    :return: zip
    """

    # Determine the number of hits >= each threshold in the experimental and control data
    experimental_count = []
    for i in range(len(thresholds)):
        count = 0
        for j in range(len(experiment_data)):
            if experiment_data[j] >= thresholds[i]:
                count += 1
        experimental_count.append(count)

    control_count = []
    for i in range(len(thresholds)):
        count = 0
        for j in range(len(control_data)):
            if control_data[j] >= thresholds[i]:
                count += 1
        control_count.append(count)

    # Normalize counts at each threshold
    experimental_total = float(experimental_count[0])
    experimental_count_normalized = [round(experimental_count[i] / experimental_total, 10)
                                     for i in range(len(experimental_count))]

    control_total = float(control_count[0])
    control_count_normalized = [round(control_count[i] / control_total, 10)
                                for i in range(len(control_count))]

    # Calculate an FDR at each threshold
    fdr = []
    for i in range(len(thresholds)):
        if experimental_count_normalized[i] == 0:
            fdr.append(0)  # Avoid 'division by 0' error
        else:
            fdr.append(round(control_count_normalized[i] / experimental_count_normalized[i], 10))

    # Apply monotonic transformation to FDRs
    fdr_smooth = af.fdr_monotonic_transformation(copy.deepcopy(fdr))

    # Format data
    colList = af.confidence_analysis_formatting(thresholds,
                                                experimental_count,
                                                control_count,
                                                experimental_count_normalized,
                                                control_count_normalized,
                                                fdr,
                                                fdr_smooth)

    return colList


def evaluate_loocv(positive_data, negative_data, thresholds):

    """ FUNCTION DESCRIPTION ❀
    Given a list of logistic regression probabilities for putative interactions and detections in the
    left out negative control, and a list of thresholds, false discovery rates (FDRs) are estimated at
    each threshold. The number of logistic regression probabilities belonging to putative interactions
    and detections in the left out negative control are determined and normalized at each threshold in
    order to estimate FDRs.

    :param positive_data: normalized count
    :type positive_data: nested list
    :param negative_data: normalized count of the left out negative control
    :type negative_data: nested list
    :param thresholds: range and resolution at which to evaluate probabilities
    :type thresholds: list
    :return: zip
    """

    positive_data_transposed = af.transpose_dataset(positive_data)
    negative_data_transposed = af.transpose_dataset(negative_data)

    # Sum normalized counts at each threshold
    positive_data_summed = [round(sum(positive_data_transposed[i]), 10)
                            for i in range(1, len(positive_data_transposed))]
    negative_data_summed = [round(sum(negative_data_transposed[i]), 10)
                            for i in range(1, len(negative_data_transposed))]

    # Estimate a false discovery rate at each probability threshold
    fdr = []
    for i in range(len(positive_data_summed)):
        fdr.append(round(negative_data_summed[i] / positive_data_summed[i], 10))

    # Apply monotonic transformation to FDRs
    fdr_smooth = af.fdr_monotonic_transformation(copy.deepcopy(fdr))

    # Format data
    colList = af.loocv_analysis_formatting(thresholds,
                                           positive_data_summed,
                                           negative_data_summed,
                                           fdr,
                                           fdr_smooth)

    return colList
