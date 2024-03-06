#!/usr/bin/env python3
# Emily Hashimoto-Roth ☆
# Lavallée-Adam Lab
# Model objects for analysis and machine learning training
# Python 3.7


import copy
import math
import statistics
import Utilities as af


class Analyze:

    def __init__(self):
        self.bait = []
        self.preys = []
        self.spectralCounts = []
        self.sum = 0
        self.normalizedList = []
        self.controls = []
        self.averages = []
        self.standardDeviations = []
        self.zscoreList = []
        self.foldchangeList = []

    # Add preys identified with non-zero spectral count data
    def addPreys(self, experiment, preys_all):

        """ FUNCTION DESCRIPTION ❀
        Creates two lists for a given experiment, containing the names of all its detected preys
        and their respective spectral counts, separately.

        :param experiment: experiment bait name (becomes identifier for object)
        :type experiment: str
        :param preys_all: all preys detected
        :type preys_all: list
        """

        self.bait = experiment[0]
        spectral_count_data = experiment[1:]

        for i in range(len(spectral_count_data)):
            if spectral_count_data[i] != 0:
                self.spectralCounts.append(spectral_count_data[i])
                self.preys.append(preys_all[i])

    # Normalize the spectral count data
    def normalize(self):

        """ FUNCTION DESCRIPTION ❀
        Normalizes the spectral count data of a given experiment, by the sum of all detected
        spectral counts.
        """

        self.sum = sum(self.spectralCounts)
        self.normalizedList = [round(self.spectralCounts[i] / self.sum, 10)
                               for i in range(len(self.spectralCounts))]

    # Add control data corresponding to identified preys
    def addControls(self, experiment, controlData):

        """ FUNCTION DESCRIPTION ❀
        Creates a nested list for the spectral count data across all controls corresponding to the preys
        detected in a given experiment or control.

        :param experiment: bait name and spectral counts (includes zero-counts)
        :type experiment: list
        :param controlData: all Analyze objects for the controls
        :type controlData: list
        """

        # Get indices for preys identified in both the given experiment and across all controls
        spectral_count_data = experiment[1:]  # Omit bait name
        index_list = []
        for i in range(len(spectral_count_data)):
            if spectral_count_data[i] != 0:
                index_list.append(i)

        # Create a nested list (self.controls) for the shifted normalized spectral counts across
        # all controls for corresponding identified preys
        for i in range(len(index_list)):
            prey_index = index_list[i]
            tmp = []
            for j in range(len(controlData)):
                tmp.append(controlData[j].normalizedList[prey_index])
            if len(tmp) == len(controlData):
                self.controls.append(tmp)

    # Calculate Z-scores and fold-change values
    def calculateScores(self, loop=None, loocv=False):

        """ FUNCTION DESCRIPTION ❀
        Creates two dictionaries for a given experiment. One containing its prey names (keys) and Z-scores
        (values). Another containing its prey names (keys) and fold-change values (values).

        :param loop: indicator of loop iteration (default = None); required for leave-one-out procedure
        :type loop: int
        :param loocv: indicator to declare leave-one-out procedure for calculations (default = False)
        :type loocv: boolean
        """

        # Calculate values without implementing a leave-one-out scheme; experiments, default
        if not loocv:
            for i in range(len(self.normalizedList)):
                self.averages.append(round(statistics.mean(self.controls[i]), 10))
                self.standardDeviations.append(round(statistics.stdev(self.controls[i]), 10))
                self.zscoreList.append(af.calculate_zscore(self.normalizedList[i],
                                                           statistics.mean(self.controls[i]),
                                                           statistics.stdev(self.controls[i])))
            for i in range(len(self.normalizedList)):
                self.foldchangeList.append(af.calculate_foldchange(self.normalizedList[i],
                                                                   statistics.mean(self.controls[i])))

        # Calculate values while implementing a leave-one-out scheme; controls
        if loocv:
            # Create copy of control data
            # Remove value corresponding to loocv run (loop value = loocv run)
            # i.e., Flag control is the first to be removed in the loocv
            #       Value corresponding to Flag is at position 0 of each nested list in the control data
            #       In the first iteration through the loop, every element at position 0 in each nested
            #       list is removed (popped)
            # Control data is restored at the beginning of each iteration
            controls_copy = copy.deepcopy(self.controls)
            removed_values = [controls_copy[i].pop(loop) for i in range(len(controls_copy))]

            for i in range(len(self.normalizedList)):
                self.averages.append(round(statistics.mean(controls_copy[i]), 10))
                self.standardDeviations.append(round(statistics.stdev(controls_copy[i]), 10))
                self.zscoreList.append(af.calculate_zscore(self.normalizedList[i],
                                                           statistics.mean(controls_copy[i]),
                                                           statistics.stdev(controls_copy[i])))
            for i in range(len(self.normalizedList)):
                self.foldchangeList.append(af.calculate_foldchange(self.normalizedList[i],
                                                                   statistics.mean(controls_copy[i])))

    def calculateScoresV2(self, loop=None, loocv=False):

        # Calculate values without implementing a leave-one-out scheme; experiments, default
        if not loocv:
            for i in range(len(self.normalizedList)):
                self.averages.append(round(statistics.mean(self.controls[i]), 10))
                self.standardDeviations.append(round(statistics.stdev(self.controls[i]), 10))
                self.zscoreList.append(af.calculate_zscore(self.normalizedList[i],
                                                           statistics.mean(self.controls[i]),
                                                           statistics.stdev(self.controls[i])))
            for i in range(len(self.normalizedList)):
                self.foldchangeList.append(af.calculate_foldchange(self.normalizedList[i],
                                                                   statistics.mean(self.controls[i])))

        # Calculate values while implementing a leave-one-out scheme; controls
        if loocv:
            for i in range(len(self.normalizedList)):
                # Calculate updated mean
                updated_mean = round((sum(self.controls[i]) - self.controls[i][loop]) / (len(self.controls[i]) - 1), 10)
                self.averages.append(updated_mean)
                # Calculate updated standard deviation (s1=step1, s2=step2, s3=step3)
                updated_stdev_s1 = [round(abs(self.controls[i][j] - updated_mean) ** 2, 10)
                                    for j in range(len(self.controls[i]))]
                updated_stdev_s2 = round(sum(updated_stdev_s1) - (abs(self.controls[i][loop] - updated_mean) ** 2), 10)
                updated_stdev_s3 = round(abs(updated_stdev_s2 / (len(self.controls[i]) - 1)), 10)  # - 2 for sample??
                updated_stdev = round(math.sqrt(updated_stdev_s3), 10)
                self.standardDeviations.append(updated_stdev)
                # Calculate Z-score
                self.zscoreList.append(af.calculate_zscore(self.normalizedList[i],
                                                           updated_mean,
                                                           updated_stdev))
                # Calculate fold-change
                self.foldchangeList.append(af.calculate_foldchange(self.normalizedList[i],
                                                                   updated_mean))


class Build:

    def __init__(self, analyze_object):
        self.analyze = analyze_object  # Analyze object for a given experiment or control
        self.bait_preys = []  # Concatenated bait-prey pair detections for a given experiment or control
        self.feature1 = []  # Normalized spectral count
        self.feature2 = []  # Average spectral count in the controls
        self.feature3 = []  # Standard deviation of spectral count in the controls
        self.feature4 = []  # Control max of spectral count in the controls
        self.feature5 = []  # Fold-change relative to spectral count in the controls
        self.labels = []  # 1: Interaction; 0: Non-specific-binding

    # Add bait-prey identifications
    def startBuild(self):

        """ FUNCTION DESCRIPTION ❀
        Creates a list of bait-prey identifications for a given experiment.

        # :param analyze_object: object created for a given experiment or control
        # :type analyze_object: Analyze
        """

        bait = self.analyze.bait
        preys = self.analyze.preys
        self.bait_preys = [str(bait) + '+' + str(preys[i]) for i in range(len(preys))]

    # Add normalized spectral count data
    def addSpecCount(self):

        """ FUNCTION DESCRIPTION ❀
        Creates a list for the feature containing the standardized spectral count data for a given
        experiment.
        """

        # self.feature1 = analyze_object.normalizedList
        self.feature1 = self.analyze.normalizedList

    # Engineer features that require calculations relative to control data
    def engineerFeatures_Labels(self, loop, threshold, loocv=False, control=False):

        """ FUNCTION DESCRIPTION ❀
        Creates a list for the features containing the mean, standard deviation, maximum normalized
        spectral count value, fold-change values, and Z-scores corresponding to the prey identifications
        of a given experiment or control.

        :param loop: indicator of loop iteration (default = None); required for leave-one-out procedure
        :type loop: int
        :param threshold: threshold for which examples are labeled "Interactions"
        :type threshold: float
        :param loocv: indicator to declare leave-one-out procedure for calculations (default = False)
        :type loocv: boolean
        :param control: indicator to declare whether iteration corresponds to a control experiment (default = False)
        :type control: boolean
        """

        if not loocv:
            controls_copy = copy.deepcopy(self.analyze.controls)
            self.feature2 = [round(statistics.mean(controls_copy[i]), 10) for i in range(len(self.feature1))]
            self.feature3 = [round(statistics.stdev(controls_copy[i]), 10) for i in range(len(self.feature1))]
            self.feature4 = [max(controls_copy[i]) for i in range(len(self.feature1))]
            self.feature5 = [af.calculate_foldchange(self.feature1[i],
                                                     statistics.mean(controls_copy[i]))
                             for i in range(len(self.feature1))]
            loocv_zscores = [af.calculate_zscore(self.feature1[i],
                                                 statistics.mean(controls_copy[i]),
                                                 statistics.stdev(controls_copy[i]))
                             for i in range(len(self.feature1))]

            # Assign labels based on newly computed Z-scores
            if not control:
                for j in range(len(loocv_zscores)):
                    if loocv_zscores[j] >= threshold:
                        self.labels.append('Interaction')
                    else:
                        self.labels.append('Non-specific-binding')
            if control:
                for j in range(len(loocv_zscores)):
                    self.labels.append('Non-specific-binding')

        if loocv:
            controls_copy = copy.deepcopy(self.analyze.controls)
            removed_values = [controls_copy[i].pop(loop) for i in range(len(controls_copy))]
            self.feature2 = [round(statistics.mean(controls_copy[i]), 10) for i in range(len(self.feature1))]
            self.feature3 = [round(statistics.stdev(controls_copy[i]), 10) for i in range(len(self.feature1))]
            self.feature4 = [max(controls_copy[i]) for i in range(len(self.feature1))]
            self.feature5 = [af.calculate_foldchange(self.feature1[i],
                                                     statistics.mean(controls_copy[i]))
                             for i in range(len(self.feature1))]
            loocv_zscores = [af.calculate_zscore(self.feature1[i],
                                                 statistics.mean(controls_copy[i]),
                                                 statistics.stdev(controls_copy[i]))
                             for i in range(len(self.feature1))]

            # Assign labels based on newly computed Z-scores
            if not control:
                for j in range(len(loocv_zscores)):
                    if loocv_zscores[j] >= threshold:
                        self.labels.append('Interaction')
                    else:
                        self.labels.append('Non-specific-binding')
            if control:
                for j in range(len(loocv_zscores)):
                    self.labels.append('Non-specific-binding')
