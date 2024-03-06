#!/usr/bin/env python3
# Emily Hashimoto-Roth ☆
# Lavallée-Adam Lab
# MAGPIE: Machine learning Assessment with loGistic regression of Protein-protein IntEractions
# Model using normalized spectral count data
# Python 3.7


import os
import sys
import shutil
import csv
import copy
import random
import Utilities as af
import analysisClasses as ac
import analysisEvaluate as ae
import Model as mod


def main():
    # Import data for analysis
    experiment_file = str(sys.argv[1])
    control_file = str(sys.argv[2])
    user_threshold = float(sys.argv[3])

    # Create directory for classification, Z-score, fold-change analysis results
    # If results directory exists from previous run, overwrite
    results_dir = 'RESULTS'
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
        os.makedirs(results_dir)
    else:
        os.makedirs(results_dir)

    # Create directory for all other result files
    # If supplemental results directory exists from previous run, overwrite
    suppl_results_dir = 'SUPPL_RESULTS'
    if os.path.exists(suppl_results_dir):
        shutil.rmtree(suppl_results_dir)
        os.mkdir(suppl_results_dir)
    else:
        os.mkdir(suppl_results_dir)

    print('\nLoading input data...')

    # Import experimental data
    experimental_data = []
    with open(experiment_file, 'r') as f1:
        content = csv.reader(f1)
        for i in content:
            experimental_data.append(i)
    experimental_baits = experimental_data[0][1:]
    transposed_experimental_data = af.transpose_dataset(experimental_data)
    experimental_preys_all = transposed_experimental_data[0][1:]

    # Format experimental data
    for i in range(1, len(transposed_experimental_data)):
        for j in range(1, len(transposed_experimental_data[i])):
            transposed_experimental_data[i][j] = float(transposed_experimental_data[i][j])
    experimental_spectral_count = transposed_experimental_data[1:]

    # Import control data
    control_data = []
    with open(control_file, 'r') as f2:
        content = csv.reader(f2)
        for i in content:
            control_data.append(i)
    control_baits = control_data[0][1:]
    transposed_control_data = af.transpose_dataset(control_data)
    control_preys_all = transposed_control_data[0][1:]

    # Format control data
    for i in range(1, len(transposed_control_data)):
        for j in range(1, len(transposed_control_data[i])):
            transposed_control_data[i][j] = float(transposed_control_data[i][j])
    control_spectral_count = transposed_control_data[1:]

    # Add pseudo-count to control data
    control_spectral_pseudo_count = []
    control_spectral_count_copy = copy.deepcopy(control_spectral_count)
    for i in range(len(control_spectral_count_copy)):
        control_spectral_pseudo_count.append(af.add_pseudo_count(control_spectral_count_copy[i]))

    # Save for reference
    control_preys_pseudo = copy.deepcopy(control_preys_all)
    control_preys_pseudo.insert(0, 'Preys')
    control_spectral_count_with_pseudo_count = af.transpose_dataset(control_spectral_pseudo_count.copy())
    for i in range(len(control_spectral_count_with_pseudo_count)):
        control_spectral_count_with_pseudo_count[i].insert(0, control_preys_pseudo[i])
    with open(f'SUPPL_RESULTS/controls_with_pseudo_count.csv', 'w+') as f3:
        writer = csv.writer(f3, delimiter=',')
        for i in range(len(control_spectral_count_with_pseudo_count)):
            writer.writerow(control_spectral_count_with_pseudo_count[i])

    # Create an Analyze object for each experiment; store objects in a list
    experiment_analysis = []
    for i in range(len(experimental_baits)):
        experiment = ac.Analyze()
        experiment.addPreys(experimental_spectral_count[i], experimental_preys_all)
        experiment.normalize()
        experiment_analysis.append(experiment)

    # Create an Analyze object for each control; store objects in a list
    control_analysis_with_pseudo_count = []
    for i in range(len(control_baits)):
        control = ac.Analyze()
        control.addPreys(control_spectral_pseudo_count[i], control_preys_all)
        control.normalize()
        control_analysis_with_pseudo_count.append(control)

    # Calculate Z-score and fold-change values for each experiment
    # Save data for each experiment
    for i in range(len(experiment_analysis)):
        experiment_analysis[i].addControls(experimental_spectral_count[i],
                                           control_analysis_with_pseudo_count)
        experiment_analysis[i].calculateScores(loop=None, loocv=False)
        name = experiment_analysis[i].bait

        col1 = copy.deepcopy(experiment_analysis[i].preys)  # maybe swap for shallow copy...?
        col1.insert(0, 'Preys')
        col2 = copy.deepcopy(experiment_analysis[i].spectralCounts)
        col2.insert(0, 'Count')
        col3 = copy.deepcopy(experiment_analysis[i].normalizedList)
        col3.insert(0, 'NormalizedCount')
        col4 = copy.deepcopy(experiment_analysis[i].averages)
        col4.insert(0, 'Average')
        col5 = copy.deepcopy(experiment_analysis[i].standardDeviations)
        col5.insert(0, 'StandardDev')
        col6 = copy.deepcopy(experiment_analysis[i].zscoreList)
        col6.insert(0, 'ZScores')
        col7 = copy.deepcopy(experiment_analysis[i].foldchangeList)
        col7.insert(0, 'FoldChanges')
        colList = zip(col1, col2, col3, col4, col5, col6, col7)
        with open(f'SUPPL_RESULTS/experiment_{name}_data.csv', 'w+') as f4:
            writer = csv.writer(f4, delimiter=',')
            for j in colList:
                writer.writerow(j)

    # Calculate Z-score and fold-change values for each control
    # Save data for each control
    iteration = -1
    for i in range(len(control_analysis_with_pseudo_count)):
        iteration += 1
        control_analysis_with_pseudo_count[i].addControls(control_spectral_pseudo_count[i],
                                                          control_analysis_with_pseudo_count)
        control_analysis_with_pseudo_count[i].calculateScores(iteration, loocv=True)
        name = control_analysis_with_pseudo_count[i].bait

        col1 = copy.deepcopy(control_analysis_with_pseudo_count[i].preys)
        col1.insert(0, 'Preys')
        col2 = copy.deepcopy(control_analysis_with_pseudo_count[i].spectralCounts)
        col2.insert(0, 'Count')
        col3 = copy.deepcopy(control_analysis_with_pseudo_count[i].normalizedList)
        col3.insert(0, 'NormalizedCount')
        col4 = copy.deepcopy(control_analysis_with_pseudo_count[i].averages)
        col4.insert(0, 'Average')
        col5 = copy.deepcopy(control_analysis_with_pseudo_count[i].standardDeviations)
        col5.insert(0, 'StandardDev')
        col6 = copy.deepcopy(control_analysis_with_pseudo_count[i].zscoreList)
        col6.insert(0, 'ZScores')
        col7 = copy.deepcopy(control_analysis_with_pseudo_count[i].foldchangeList)
        col7.insert(0, 'FoldChanges')
        colList = zip(col1, col2, col3, col4, col5, col6, col7)
        with open(f'SUPPL_RESULTS/control_{name}_data.csv', 'w+') as f5:
            writer = csv.writer(f5, delimiter=',')
            for j in colList:
                writer.writerow(j)

    ############################
    # Perform Z-score analysis #
    ############################

    print('\nPerforming Z-score analysis...')

    # Extract Z-score values from all experimental Analyze objects
    all_experimental_zscores = []
    for i in range(len(experiment_analysis)):
        all_experimental_zscores.append(copy.deepcopy(experiment_analysis[i].zscoreList))
    all_experimental_zscores_flat = [item for sublist in all_experimental_zscores for item in sublist]

    # Extract Z-score values from all control Analyze objects
    all_control_zscores = []
    for i in range(len(control_analysis_with_pseudo_count)):
        all_control_zscores.append(copy.deepcopy(control_analysis_with_pseudo_count[i].zscoreList))
    all_control_zscores_flat = [item for sublist in all_control_zscores for item in sublist]

    # Define and format Z-score thresholds
    zscore_thresholds = [i * 0.1 for i in [*range(-200, 201)]]
    zscore_thresholds_formatted = [round(i, 1) for i in zscore_thresholds]  # Round to 1 decimal

    # Evaluate Z-scores and save
    colList_zscore = ae.evaluate(all_experimental_zscores_flat,
                                 all_control_zscores_flat,
                                 zscore_thresholds_formatted)
    with open(f'RESULTS/zscore_analysis.csv', 'w+') as f6:
        writer = csv.writer(f6, delimiter=',')
        for i in colList_zscore:
            writer.writerow(i)

    ################################
    # Perform fold-change analysis #
    ################################

    print('\nPerforming fold-change analysis...')

    # Extract fold-change values from all experimental Analyze objects
    all_experimental_foldchanges = []
    for i in range(len(experiment_analysis)):
        all_experimental_foldchanges.append(copy.deepcopy(experiment_analysis[i].foldchangeList))
    all_experimental_foldchanges_flat = [item for sublist in all_experimental_foldchanges for item in sublist]

    # Extract fold-change values from all control Analyze objects
    all_control_foldchanges = []
    for i in range(len(control_analysis_with_pseudo_count)):
        all_control_foldchanges.append(copy.deepcopy(control_analysis_with_pseudo_count[i].foldchangeList))
    all_control_foldchanges_flat = [item for sublist in all_control_foldchanges for item in sublist]

    # Define and format fold-change values thresholds
    foldchange_thresholds = [i * 0.1 for i in [*range(0, 101)]]
    foldchange_thresholds_formatted = [round(i, 1) for i in foldchange_thresholds]  # Round to 1 decimal

    # Evaluate fold-change values and save
    colList_foldchange = ae.evaluate(all_experimental_foldchanges_flat,
                                     all_control_foldchanges_flat,
                                     foldchange_thresholds_formatted)
    with open(f'RESULTS/foldchange_analysis.csv', 'w+') as f7:
        writer = csv.writer(f7, delimiter=',')
        for i in colList_foldchange:
            writer.writerow(i)

    ##################################################
    # Perform leave-one-control-out cross-validation #
    ##################################################

    # Generate positive and negative examples, implementing a leave-one-control-out (LOCO) cross-validation scheme
    # n sets of positive and negative examples will be produced, such that n = the number of controls (i.e., n = k)
    # Define and format probability thresholds
    probability_thresholds = [i * 0.01 for i in [*range(0, 100)]]
    probability_thresholds_formatted = [round(i, 2) for i in probability_thresholds]  # Round to 2 decimals

    # Generate positive examples for each loco run
    loco_run = -1
    for i in range(len(control_analysis_with_pseudo_count)):
        loco_run += 1
        loco_run_name = control_analysis_with_pseudo_count[i].bait
        all_positive_examples = []
        for j in range(len(experiment_analysis)):
            # Create a Build object for the positive examples; store in a list
            # Features will be calculated, omitting the data of the control corresponding to the loco run
            # loco_run = the index of the loco run (passed to loop parameter of the Build object)
            #   i.e., 0-Flag, 1-HA, 2-LC3B, 3-METTL23, 4-RPAP2
            positive_examples = ac.Build(experiment_analysis[j])
            positive_examples.startBuild()
            positive_examples.addSpecCount()
            positive_examples.engineerFeatures_Labels(loop=loco_run, threshold=user_threshold, loocv=True)
            all_positive_examples.append(positive_examples)
        # Concatenate information from each Build object and save
        colList_positive = af.model_building_formatting_list(all_positive_examples)
        with open(f'SUPPL_RESULTS/loco_{loco_run_name}_positive_examples.csv', 'w+') as f8:
            writer = csv.writer(f8, delimiter=',')
            for k in colList_positive:
                writer.writerow(k)

    # Generate negative examples for each loco run
    loco_run = -1
    for i in range(len(control_analysis_with_pseudo_count)):
        loco_run += 1
        loco_run_name = control_analysis_with_pseudo_count[i].bait
        all_negative_examples = []
        control_analysis_copy = copy.deepcopy(control_analysis_with_pseudo_count)
        remove_run = control_analysis_copy.pop(i)
        for j in range(len(control_analysis_with_pseudo_count) - 1):
            # Create a Build object for the negative examples; store in a list
            # Features will be calculated, omitting the data of the control corresponding to the loco run
            # loco_run = the index of the loco run (passed to loop parameter of the Build object)
            #   i.e., 0-Flag, 1-HA, 2-LC3B, 3-METTL23, 4-LOOCV_RPAP2
            negative_examples = ac.Build(control_analysis_copy[j])
            negative_examples.startBuild()
            negative_examples.addSpecCount()
            negative_examples.engineerFeatures_Labels(loop=loco_run, threshold=user_threshold, loocv=True, control=True)
            all_negative_examples.append(negative_examples)
        # Concatenate information from each Build object and save
        colList_negative = af.model_building_formatting_list(all_negative_examples)
        with open(f'SUPPL_RESULTS/loco_{loco_run_name}_negative_examples.csv', 'w+') as f9:
            writer = csv.writer(f9, delimiter=',')
            for k in colList_negative:
                writer.writerow(k)

    # Generate validation examples for each loco run
    loco_run = -1
    for i in range(len(control_analysis_with_pseudo_count)):
        loco_run += 1
        loco_run_name = control_analysis_with_pseudo_count[i].bait
        false_positives = ac.Build(control_analysis_with_pseudo_count[i])
        false_positives.startBuild()
        false_positives.addSpecCount()
        false_positives.engineerFeatures_Labels(loop=loco_run, threshold=user_threshold, loocv=True, control=True)
        colList_false_positives = af.false_positive_building(false_positives)
        with open(f'SUPPL_RESULTS/loco_{loco_run_name}_validation_examples.csv', 'w+') as f10:
            writer = csv.writer(f10, delimiter=',')
            for j in colList_false_positives:
                writer.writerow(j)

    # Identify positive and negative training examples for each loco run
    for i in range(len(control_analysis_with_pseudo_count)):
        bait_name = control_analysis_with_pseudo_count[i].bait

        # Open a loco positive examples file and subset the examples labeled "Interaction"
        loco_positive_examples_with_header = []
        with open(f'SUPPL_RESULTS/loco_{bait_name}_positive_examples.csv', 'r') as f11:
            content = csv.reader(f11, delimiter=',')
            for j in content:
                loco_positive_examples_with_header.append(j)
        loco_positive_examples = loco_positive_examples_with_header[1:]
        loco_positive_training = [loco_positive_examples[p] for p in range(len(loco_positive_examples))
                                  if loco_positive_examples[p][-1] == 'Interaction']

        # Identify total number of positives training examples; multiply to change class balance
        examples_total = int(len(loco_positive_training))

        # Save positive training examples for given loco run
        loco_positive_training.insert(0, loco_positive_examples_with_header[0])
        with open(f'SUPPL_RESULTS/loco_{bait_name}_positive_training_examples.csv', 'w+') as f12:
            writer = csv.writer(f12, delimiter=',')
            for q in range(len(loco_positive_training)):
                writer.writerow(loco_positive_training[q])

        # Open the corresponding loco negative examples file and randomly sample N examples for training
        loco_negative_examples_with_header = []
        with open(f'SUPPL_RESULTS/loco_{bait_name}_negative_examples.csv', 'r') as f13:
            content = csv.reader(f13, delimiter=',')
            for j in content:
                loco_negative_examples_with_header.append(j)
        loco_negative_examples = loco_negative_examples_with_header[1:]
        loco_negative_training = random.sample(loco_negative_examples, examples_total)

        # Save negative training examples for given loco run
        loco_negative_training.insert(0, loco_negative_examples_with_header[0])
        with open(f'SUPPL_RESULTS/loco_{bait_name}_negative_training_examples.csv', 'w+') as f14:
            writer = csv.writer(f14, delimiter=',')
            for q in range(len(loco_negative_training)):
                writer.writerow(loco_negative_training[q])

    # Carry out leave-one-control-out cross-validation and estimate the model's false-discovery rate
    for i in range(len(control_analysis_with_pseudo_count)):
        bait_name = control_analysis_with_pseudo_count[i].bait
        positive_predictions, negative_predictions, loocv_weights = \
            mod.run_model(f'SUPPL_RESULTS/loco_{bait_name}_positive_training_examples.csv',
                          f'SUPPL_RESULTS/loco_{bait_name}_negative_training_examples.csv',
                          f'SUPPL_RESULTS/loco_{bait_name}_positive_examples.csv',
                          f'SUPPL_RESULTS/loco_{bait_name}_validation_examples.csv')

        # Subset positive prediction probabilities for evaluation
        positive_predictions_probabilities = []
        for j in range(1, len(positive_predictions)):
            positive_predictions_probabilities.append(positive_predictions[j][1])

        # Subset negative predictions probabilities for evaluation
        negative_predictions_probabilities = []
        for j in range(1, len(negative_predictions)):
            negative_predictions_probabilities.append(negative_predictions[j][1])

        # Evaluate predicted probabilities and save
        colList_loco_probability = ae.evaluate(positive_predictions_probabilities,
                                               negative_predictions_probabilities,
                                               copy.deepcopy(probability_thresholds_formatted))
        with open(f'SUPPL_RESULTS/loco_{bait_name}_model_results.csv', 'w+') as f15:
            writer = csv.writer(f15, delimiter=',')
            for k in colList_loco_probability:
                writer.writerow(k)

        # Save prediction files
        with open(f'SUPPL_RESULTS/loco_{bait_name}_positive_predictions.csv', 'w+') as f16:
            writer = csv.writer(f16, delimiter=',')
            for k in range(len(positive_predictions)):
                writer.writerow(positive_predictions[k])
        with open(f'SUPPL_RESULTS/loco_{bait_name}_negative_predictions.csv', 'w+') as f17:
            writer = csv.writer(f17, delimiter=',')
            for k in range(len(negative_predictions)):
                writer.writerow(negative_predictions[k])

    ##################################################
    # Estimate an overall false discovery rate (FDR) #
    ##################################################

    loco_normalized_positive_counts = []
    loco_normalized_negative_counts = []
    for i in range(len(control_analysis_with_pseudo_count)):
        bait_name = control_analysis_with_pseudo_count[i].bait
        loco_run_counts_with_header = []
        with open(f'SUPPL_RESULTS/loco_{bait_name}_model_results.csv', 'r') as f18:
            content = csv.reader(f18, delimiter=',')
            for j in content:
                loco_run_counts_with_header.append(j)
        loco_run_counts = loco_run_counts_with_header[1:]  # Omit header

        # Convert strings to floats where appropriate
        for m in range(len(loco_run_counts)):
            for n in range(len(loco_run_counts[m])):
                loco_run_counts[m][n] = float(loco_run_counts[m][n])

        # Format and append normalized putative interaction counts
        loco_run_normalized_positive_counts = [loco_run_counts[p][3] for p in range(len(loco_run_counts))]
        loco_run_normalized_positive_counts.insert(0, bait_name)
        loco_normalized_positive_counts.append(loco_run_normalized_positive_counts)

        # Format and append normalized false-positive counts
        loco_run_normalized_negative_counts = [loco_run_counts[p][4] for p in range(len(loco_run_counts))]
        loco_run_normalized_negative_counts.insert(0, bait_name)
        loco_normalized_negative_counts.append(loco_run_normalized_negative_counts)

    colList_fdr = ae.evaluate_loocv(loco_normalized_positive_counts,
                                    loco_normalized_negative_counts,
                                    copy.deepcopy(probability_thresholds_formatted))
    with open(f'RESULTS/loco_fdr_estimation.csv', 'w+') as f19:
        writer = csv.writer(f19, delimiter=',')
        for k in colList_fdr:
            writer.writerow(k)

    #############
    # Run Model #
    #############

    print('\nRunning logistic regression model...')

    # Generate positive examples
    for i in range(len(experiment_analysis)):
        run_name = experiment_analysis[i].bait
        all_positive_examples = []
        experiment_analysis_copy = copy.deepcopy(experiment_analysis)

        # Create Build object for given experiment
        positive_examples = ac.Build(experiment_analysis_copy[i])
        positive_examples.startBuild()
        positive_examples.addSpecCount()
        positive_examples.engineerFeatures_Labels(loop=None, threshold=user_threshold, loocv=False)
        all_positive_examples.append(positive_examples)

        # Concatenate information for given Build object and save
        colList_positive = af.model_building_formatting_list(all_positive_examples)
        with open(f'SUPPL_RESULTS/experiment_{run_name}_positive_examples.csv', 'w+') as f20:
            writer = csv.writer(f20, delimiter=',')
            for k in colList_positive:
                writer.writerow(k)

    # Generate negative examples
    loco_run = -1
    for i in range(len(control_analysis_with_pseudo_count)):
        loco_run += 1
        run_name = control_analysis_with_pseudo_count[i].bait
        all_negative_examples = []
        control_analysis_copy = copy.deepcopy(control_analysis_with_pseudo_count)

        # Create Build object for given control
        negative_examples = ac.Build(control_analysis_copy[i])
        negative_examples.startBuild()
        negative_examples.addSpecCount()
        negative_examples.engineerFeatures_Labels(loop=loco_run, threshold=user_threshold, loocv=True, control=True)
        all_negative_examples.append(negative_examples)

        # Concatenate information for given Build object and save
        colList_negative = af.model_building_formatting_list(all_negative_examples)
        with open(f'SUPPL_RESULTS/control_{run_name}_negative_examples.csv', 'w+') as f21:
            writer = csv.writer(f21, delimiter=',')
            for k in colList_negative:
                writer.writerow(k)

    # Identify positive training examples
    all_positive_examples = []
    all_positive_training_examples = []
    for i in range(len(experiment_analysis)):
        bait_name = experiment_analysis[i].bait

        # Open the positive examples dataset for a given experiment
        positive_examples_with_header = []
        with open(f'SUPPL_RESULTS/experiment_{bait_name}_positive_examples.csv', 'r') as f22:
            content = csv.reader(f22, delimiter=',')
            for j in content:
                positive_examples_with_header.append(j)
        positive_examples = positive_examples_with_header[1:]
        for l in range(len(positive_examples)):
            all_positive_examples.append(positive_examples[l])

        # Subset examples labeled "Interaction"
        positive_training = [positive_examples[k] for k in range(len(positive_examples))
                             if positive_examples[k][-1] == 'Interaction']
        for m in range(len(positive_training)):
            all_positive_training_examples.append(positive_training[m])
    positive_examples_total = int(len(all_positive_training_examples))

    # Identify negative training examples
    all_negative_examples = []
    for i in range(len(control_analysis_with_pseudo_count)):
        bait_name = control_analysis_with_pseudo_count[i].bait

        # Open the negative examples dataset for a given control
        negative_examples_with_header = []
        with open(f'SUPPL_RESULTS/control_{bait_name}_negative_examples.csv', 'r') as f23:
            content = csv.reader(f23, delimiter=',')
            for j in content:
                negative_examples_with_header.append(j)
        negative_examples = negative_examples_with_header[1:]

        # Concatenate all negative examples
        for l in range(len(negative_examples)):
            all_negative_examples.append(negative_examples[l])

    # Random sample N negative examples for training
    # N = positive_examples_total
    random.seed(0)
    all_negative_training_examples = random.sample(all_negative_examples, positive_examples_total)

    # Format training datasets and save
    column_names = ['BaitPrey', 'StandardizedCount', 'ControlAverage', 'ControlStandardDeviation',
                    'ControlMax', 'FoldChange', 'Class']
    all_positive_examples.insert(0, column_names)
    all_positive_training_examples.insert(0, column_names)
    all_negative_examples.insert(0, column_names)
    all_negative_training_examples.insert(0, column_names)
    with open(f'SUPPL_RESULTS/model_positive_examples.csv', 'w+') as f24:
        writer = csv.writer(f24, delimiter=',')
        for i in range(len(all_positive_examples)):
            writer.writerow(all_positive_examples[i])
    with open(f'SUPPL_RESULTS/model_positive_training_examples.csv', 'w+') as f25:
        writer = csv.writer(f25, delimiter=',')
        for i in range(len(all_positive_training_examples)):
            writer.writerow(all_positive_training_examples[i])
    with open(f'SUPPL_RESULTS/model_negative_examples.csv', 'w+') as f26:
        writer = csv.writer(f26, delimiter=',')
        for i in range(len(all_negative_examples)):
            writer.writerow(all_negative_examples[i])
    with open(f'SUPPL_RESULTS/model_negative_training_examples.csv', 'w+') as f27:
        writer = csv.writer(f27, delimiter=',')
        for i in range(len(all_negative_training_examples)):
            writer.writerow(all_negative_training_examples[i])

    # Train and run model
    model_positive_predictions, model_negative_predictions, model_weights = \
        mod.run_model(f'SUPPL_RESULTS/model_positive_training_examples.csv',
                      f'SUPPL_RESULTS/model_negative_training_examples.csv',
                      f'SUPPL_RESULTS/model_positive_examples.csv',
                      f'SUPPL_RESULTS/model_negative_examples.csv')

    # Format model weights
    model_weights_formatted = af.format_weights(model_weights)

    # Identify high-confidence interactions
    # Putative interactions whose probability of being a true interaction >= 0.99
    high_confidence_interactions = []
    for i in range(1, len(model_positive_predictions)):
        if model_positive_predictions[i][1] >= 0.99:
            high_confidence_interactions.append(model_positive_predictions[i])
    high_confidence_interactions.insert(0, ['Input', 'Probability_Interaction', 'Probability_Non_Specific_Binding'])

    # Format master output file for all putative protein-protein interactions
    all_positive_examples_no_header = all_positive_examples[1:]
    model_positive_predictions_no_header = model_positive_predictions[1:]
    final_output = []
    for i in range(len(all_positive_examples_no_header)):
        putative_interaction = all_positive_examples_no_header[i]
        putative_interaction.insert(-1, all_experimental_zscores_flat[i])
        putative_interaction.append(model_positive_predictions_no_header[i][1])
        final_output.append(putative_interaction)
    final_output.insert(0, ['PutativeInteraction', 'Feature1-SpectralCount', 'Feature2-AvgSpectralCount',
                            'Feature3-StnDevSpectralCount', 'Feature4-MaxSpectralCount', 'Feature5-FoldChange', 'ZScore',
                            'ClassLabel', 'Probability'])

    # Add column associating logistic regression probability of putative protein-protein interaction
    # to false discovery rate
    loco_fdr_with_header = []
    with open(f'RESULTS/loco_fdr_estimation.csv', 'r') as f28:
        content = csv.reader(f28, delimiter=',')
        for i in content:
            loco_fdr_with_header.append(i)
    loco_fdr = loco_fdr_with_header[1:]
    
    # Create probability bins
    forBins = copy.deepcopy(probability_thresholds_formatted)
    bins = []
    for i in range(len(forBins) - 1):
        minimum = forBins[i]
        maximum = forBins[i + 1]
        bins.append([minimum, maximum, float(loco_fdr[i][-1])])
    bins.append([0.99, 1.01, float(loco_fdr[-1][-1])])

    # Add false discovery rates associated to each putative interaction's probability
    for i in range(len(bins)):
        for j in range(1, len(final_output)):
            tmp = float(final_output[j][8])
            if bins[i][0] <= tmp < bins[i][1]:
                final_output[j].append(bins[i][2])
                j += 1
            else:
                j += 1

    # Add column header
    final_output[0].append('FDR')

    # Save
    # with open(f'RESULTS/model_high_confidence_interactions.csv', 'w+') as f28:
    #     writer = csv.writer(f28, delimiter=',')
    #     for i in range(len(high_confidence_interactions)):
    #         writer.writerow(high_confidence_interactions[i])
    with open(f'SUPPL_RESULTS/model_positive_predictions.csv', 'w+') as f29:
        writer = csv.writer(f29, delimiter=',')
        for i in range(len(model_positive_predictions)):
            writer.writerow(model_positive_predictions[i])
    with open(f'SUPPL_RESULTS/model_negative_predictions.csv', 'w+') as f30:
        writer = csv.writer(f30, delimiter=',')
        for i in range(len(model_negative_predictions)):
            writer.writerow(model_negative_predictions[i])
    with open(f'RESULTS/feature_weights.txt', 'w+') as f31:
        writer = csv.writer(f31, delimiter='\t')
        for i in range(len(model_weights_formatted)):
            writer.writerow(model_weights_formatted[i])
    with open(f'RESULTS/magpie_results.csv', 'w+') as f32:
        writer = csv.writer(f32, delimiter=',')
        for i in range(len(final_output)):
            writer.writerow(final_output[i])

    print('\nDone!\n')


if __name__ == "__main__":
    main()
