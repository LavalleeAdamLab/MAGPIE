###########################
# By Emily Hashimoto-Roth #
#       MAGPIE v1.0       #
###########################


CONTENTS OF THIS FILE
---------------------

 * Introduction
 * Dependencies
 * Running MAGPIE
 * Input files
 * Output files


INTRODUCTION
------------

MAGPIE is a machine learning-based analysis tool for the confidence assessment of protein-protein interactions (PPIs) in human plasma samples, identified by immunoprecipitation coupled to tandem mass spectrometry (IP-MS/MS). This algorithm constructs a class-balanced training dataset of likely high-confidence putative PPIs and examples of non-specific protein binding. A logistic regression classification model is trained and used to assign probabilities of being a bona fide interaction to putative PPIs. The performance of this classification is evaluated by implementing a leave-one-control-out (LOCO) cross-validation strategy to compute false discovery rates for the assigned probabilities. MAGPIE was tested using spectral count data from ten IP-MS/MS produced in Dr. Benoit Coulombe's lab at the Institut de recherches cliniques de Montréal (IRCM).


DEPENDENCIES
------------

 * Python version 3.7+ (download: https://www.python.org/downloads/)
 * Scikit-learn version 0.24.0+ (download: https://scikit-learn.org/stable/install.html)


RUNNING MAGPIE
--------------

1. Download the MAGPIE package.
	1.1 Save the package to your desired location (e.g., your home directory).
	1.2 Use the path, accordingly, to access the package.
2. Prepare your input files as described below.
	2.1 Save your input files to the same directory as magpie.py.
3. Open a terminal (Mac/linux) or command prompt (Windows).
4. Navigate to where the MAGPIE package is stored.
5. Run the command:

	python3 magpie.py [experiment_file] [control_file] [Z-score_threshold]

Note: The specified Z-score threshold will be used to label the likely high-confidence PPIs to be used as positive training examples. We recommend starting with a Z-score threshold of 3.


INPUT FILES
-----------

MAGPIE requires two .CSV files as input, one for experimental data and one for negative control data, BOTH with the following formatting:

 * Row 1: Names of proteins targeted by IP antibodies, preceded by a column name for the protein purifications, such as "Preys"
 * Column 1: Names of proteins purified in the IP-MS/MS runs
 * Column 2 and on: Spectral count quantification of purified proteins in a given IP-MS/MS run (non-normalized, as MAGPIE normalizes the counts as per its methods)

Toy example:

	 Preys | Bait 1 | Bait 2 | Bait 3 ... Bait n
	--------------------------------- ... -------
	Prey 1 |  1032  |  1001  |  1100  ...  1270
	--------------------------------- ... -------
	Prey 2 |  876   |  786   |  897   ...  658
	--------------------------------- ... -------
	Prey 3 |  68    |  75    |  232   ...  139
	  ...     ...      ...      ...   ...  ... 
	Prey m |  36    |  126   |  24    ...  63


OUTPUT FILES
------------

MAGPIE outputs the following files to a directory called RESULTS:

 * magpie_results.csv - Results of logistic regression classification
 * loco_fdr_estimation.csv - Computed false discovery rates associated to probabilities assigned by logistic regression classification
 * features_weights.txt - Weights (coefficients) computed for each feature by optimizing the logistic regression model
 * zscore_analysis.csv - Results of discriminating putative PPIs from non-specific binding based on Z-score calculations (used for benchmarking evaluation against machine learning model)
 * foldchange_analysis.csv - Results of discriminating putative PPIs from non-specific binding based on fold-change values (used for benchmarking evaluation against machine learning model)

MAGPIE outputs the following files to a directory called SUPPL_RESULTS:

 * experiment_[IP RUN NAME]_data.csv - Data computed for Z-score and fold-change analyses for given experimental IP run
 * experiment_[IP RUN NAME]_positive_examples.csv - Feature values and classification labels computed for a given experimental IP run

 * controls_with_pseudo_count.csv - Spectral count data from negative control experiments with pseudo-count values added
 * control_[IP RUN NAME]_data.csv - Data computed for Z-score and fold-change analyses for a given negative control IP run
 * control_[IP RUN NAME]_negative_examples.csv - Feature values and classification labels computed for given negative control IP run

 * loco_[IP RUN NAME]_model_results.csv - LOCO cross-validation results having left out a given negative control IP run
 * loco_[IP RUN NAME]_positive_examples.csv - Testing data for putative PPIs, re-computed without a given negative control IP run
 * loco_[IP RUN NAME]_negative_examples.csv - Testing data for protein purifications in the negative controls, re-computed without a given negative control IP run
 * loco_[IP RUN NAME]_validation_examples.csv - Validation data for protein purifications in the left out negative control IP run
 * loco_[IP RUN NAME]_positive_training_examples.csv - Training data for putative PPIs, re-computed without a given negative control IP run
 * loco_[IP RUN NAME]_negative_training_examples.csv - Training data for protein purifications in the negative controls, re-computed without a given negative control IP run
 * loco_[IP RUN NAME]_positive_predictions.csv - Probabilities associated to putative PPIs
 * loco_[IP RUN NAME]_negative_predictions.csv - Probabilities associated to protein purifications in the left out negative control IP run

 * model_positive_examples.csv - Testing data for putative PPIs
 * model_negative_examples.csv - Testing data for detections in the negative control IP runs
 * model_positive_training_examples.csv - Training data for likely high-confidence PPIs
 * model_negative_training_examples.csv - Training data for protein purifications in the negative controls
 * model_positive_predictions.csv - Probabilities associated to putative PPIs
 * model_negative_predictions.csv - Probabilities associated to protein purifications in the negative controls




