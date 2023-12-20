import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import KDTree
from scipy.stats import ttest_ind

from helpers.funcs import *
from helpers.plotters import *

warnings.filterwarnings('ignore')

# Ensure reproducability
seed = 42
prob_thresh = 0.9

#  Features to extract from dataframe
desired_variables = ['stay_id',
                     'biocarbonate',
                     'bloodOxygen',
                     'bloodPressure',
                     'bun',
                     'creatinine',
                     'fio2',
                     'haemoglobin',
                     'heartRate',
                     'motorGCS',
                     'eyeGCS',
                     'potassium',
                     'respiratoryRate',
                     'sodium',
                     'Temperature [C]',
                     'verbalGCS',
                     'age',
                     'gender',
                     'hours_since_admission',
                     'RFD']

# For the case of passing to DiCE, change data variables to categorical type
categorical_columns = ['motorGCS',
                       'verbalGCS',
                       'eyeGCS',
                       'gender']


def obtain_stay_id_individuals(
        data,
        labels,
        label_value):
    '''
    Function used to obtain patients with a specific outcome of interest
    input
        data - patient data
        labels - data labels
        label_value - value to choose specific type of patient of interest, eg negative outcomes
    return
        full_test_df_stay_id_data - patients who have the stay id of a particular outcome

    '''

    # Filtering the stay IDs for the data of interest based off the RFD flag
    positive_stay_id = labels[labels['RFD'] == label_value]

    full_test_df_stay_id = pd.concat(
        [data, labels[['RFD', 'stay_id', 'hours_since_admission']]], axis=1)
    # Based on chatgpt
    full_test_df_stay_id_data = full_test_df_stay_id[full_test_df_stay_id['stay_id'].isin(
        positive_stay_id['stay_id'])]

    return full_test_df_stay_id_data


def train_classifier(filepath):
    '''
    Function to preprocess the data and train a classifier
    input: filepath -filepath of csv which contains the different values
    return: classifier - trained classifier
            balanced_x_train_norm - balanced training data
            balanced_y_train - balanced training labels
            x_test_norm - normalized (unbalanced ) test data
            y_test - testing labels
    '''
    # Get the variables of interest and fill in the misssing values with a mean and mode value
    filtered_df_data = initial_icu_processing(filepath, desired_variables)

    train_data, test_data = train_test_split(
        filtered_df_data, test_size=0.2, random_state=seed, shuffle=False)

    # Remove columns which are important in general but not for the training of the classifier'

    y_train = train_data[['stay_id', 'RFD', 'hours_since_admission']]
    x_train = train_data.drop(
        columns=['stay_id', 'RFD', 'hours_since_admission'])

    y_test = test_data[['stay_id', 'RFD', 'hours_since_admission']]
    x_test = test_data.drop(
        columns=['stay_id', 'RFD', 'hours_since_admission'])
    # Normalize the data
    x_train_norm, df_mean, df_std = normalize_data(x_train)
    x_test_norm = (x_test-df_mean)/df_std
    # full_norm = (filtered_df_data-df_mean)/df_std # DEL?

    # for KDTree
    ind_rfd_1 = x_train_norm[y_train['RFD'] == 1]
    ind_rfd_2 = x_train_norm[y_train['RFD'] == 2]

    tree_1 = KDTree(ind_rfd_1)
    tree_2 = KDTree(ind_rfd_2)

    # Balance the data points belonging to the different classes in the train and test set
    balanced_x_train_norm, balanced_y_train = balancing_classes(
        x_train_norm, y_train)
    balanced_x_test_norm, balanced_y_test = balancing_classes(
        x_test_norm, y_test)
    # Train classifier
    classifier = MLPClassifier(hidden_layer_sizes=(10, 10), random_state=seed, max_iter=10).fit(
        balanced_x_train_norm, balanced_y_train['RFD'])
    # Evaluate classifer performance
    train_predictions = classifier.predict(balanced_x_train_norm)
    f1_train_score = f1_score(
        balanced_y_train['RFD'], train_predictions, average='macro')
    accuracy_train_score = accuracy_score(
        balanced_y_train['RFD'], train_predictions)
    test_predictions = classifier.predict(balanced_x_test_norm)
    f1_test_score = f1_score(
        balanced_y_test['RFD'], test_predictions, average='macro'),
    accuracy_test_score = accuracy_score(
        balanced_y_test['RFD'], test_predictions)

    print('Model training complete. Metrics: training set f1',
          f1_train_score, 'train accuracy', accuracy_train_score)
    print('Balanced testing set f1', f1_test_score,
          ', accuracy', accuracy_test_score)

    # Evaluate classifer performance - original unbalanced test set
    unbalanced_test_predictions = classifier.predict(x_test_norm)
    f1_unbalanced_test_score = f1_score(
        y_test['RFD'], unbalanced_test_predictions, average='macro')
    accuracy_unbalanced_test_score = accuracy_score(
        y_test['RFD'], unbalanced_test_predictions)

    print('Unbalanced testing set f1', f1_unbalanced_test_score,
          ', accuracy', accuracy_unbalanced_test_score)

    return classifier, balanced_x_train_norm, balanced_y_train, x_test_norm, y_test, tree_1, tree_2, ind_rfd_1, ind_rfd_2


def generate_dice_cf_global(filepath,
                            num_cases_to_assess=3,
                            trace_lambda=0.9,
                            num_cfs=1,
                            oracle_desirable_cf=False,
                            ):
    '''
    Generate plot for a subset of the individuals

    Inputs:
        filepath
        num_cases_to_assess:
            How many stay_ids to process
        trace_lambda:
            Weighting between angle and proximity to counterfactual
        num_cfs:
            Number of counterfactual examples to compute TraCE against
            (for both desirable and undesirable components)
        oracle_desirable_cf:
            If wanting to use the patient's own final timepoint as the counterfactual

    Returns:
        TraCE scores
    '''
    trained_model, processed_balanced_x_train_norm, balanced_y_train, processed_x_test_norm, y_test, tree_1, tree_2, ind_rfd_1, ind_rfd_2 = train_classifier(
        filepath)
    # Join the labels to the data
    full_df = pd.concat([processed_balanced_x_train_norm,
                        balanced_y_train['RFD']], axis=1)
    full_test_df = pd.concat([processed_x_test_norm, y_test['RFD']], axis=1)
    full_df = pd.concat([full_df, full_test_df], axis=0)

    # Don't need the following two lines when using KDtree instead of DiCE
    # exp_genetic_mimic, features_to_vary = cf_generator(full_df, trained_model)
    # features_to_vary.remove('gender')

    # Includes the patient and stay ID of an individual
    positive_stay_id_patients = obtain_stay_id_individuals(
        processed_x_test_norm, y_test, 1)
    negative_stay_id_patients = obtain_stay_id_individuals(
        processed_x_test_norm, y_test, 2)

    print('--- INITIATING FOR POSITIVE OUTCOME(S) - SUCCESSFUL DISCHARGE TO HOME ---')
    print(
        f'Analysing  {num_cases_to_assess}/{len(positive_stay_id_patients.stay_id.unique())} available cases')

    positive_patient_scores, positive_times = calculate_TraCE_scores(
        positive_stay_id_patients,
        num_cases_to_assess=num_cases_to_assess,
        trace_lambda=trace_lambda,
        num_cfs=num_cfs,
        oracle_desirable_cf=oracle_desirable_cf,
        label='pos', model=trained_model,
        tree_1=tree_1, tree_2=tree_2,
        ind_rfd_1=ind_rfd_1,
        ind_rfd_2=ind_rfd_2)
    print('------------ INITIATING FOR NEGATIVE OUTCOME(S) - MORTALITY -------------')
    print(
        f'Analysing  {num_cases_to_assess}/{len(negative_stay_id_patients.stay_id.unique())} available cases')

    negative_patient_scores, negative_times = calculate_TraCE_scores(
        negative_stay_id_patients,
        num_cases_to_assess=num_cases_to_assess,
        trace_lambda=trace_lambda,
        num_cfs=num_cfs,
        oracle_desirable_cf=oracle_desirable_cf,
        label='neg',
        model=trained_model,
        tree_1=tree_1, tree_2=tree_2,
        ind_rfd_1=ind_rfd_1,
        ind_rfd_2=ind_rfd_2)

    # Save the pickle files
    positive_patient_score_file_name = 'positive_patient_scores.pkl'
    with open(positive_patient_score_file_name, 'wb') as file:
        pickle.dump(positive_patient_scores, file)

    negative_patient_score_file_name = 'negative_patient_scores.pkl'
    with open(negative_patient_score_file_name, 'wb') as file:
        pickle.dump(negative_patient_scores, file)

    # Obtains the values of interest for processing
    positive_mean_score, positive_std_score, pos_outcome_means = analysis(
        positive_patient_score_file_name, cumulative=False)
    negative_mean_score, negative_std_score, neg_outcome_means = analysis(
        negative_patient_score_file_name, cumulative=False)

    # perform Welch's t-test
    print("Welch's t-test outcome:", ttest_ind(
        pos_outcome_means, neg_outcome_means, equal_var=False))

    return (positive_mean_score, positive_std_score), (negative_mean_score, negative_std_score)


def calculate_TraCE_scores(
        patient_data,
        model,
        trace_lambda,
        num_cases_to_assess=3,
        num_cfs=3,
        oracle_desirable_cf=False,
        label='',
        tree_1='',
        tree_2='',
        ind_rfd_1='',
        ind_rfd_2='',
):
    '''
    Function used to generate the traCE scores
    inputs:
        patient_data: dataframe of the patients
            (usually linked to certain class e.g all patients ready for discharged)
        model: trained model to calculate prediction probabilities
        trace_lambda: weighting between angle (R1) and proximity (R2)
        num_cases_to_assess: randomly selecting this number of cases from the test set
        oracle_desirable_cf: utilises case's actual final timepoint as the desired counterfactual, not recommended beyond development
        tree_1: KDTree for class 1 (successful discharge)
        tree_2: KDTree fOR class 2 (mortality)
        ind_rfd_1: corpus of known outcomes from the training set for class 1 (successful discharge)
        ind_rfd_2: corpus of known outcomes from the training set for class 2 (mortality)
    return: 
        collated_scores: scores for all the different timesteps for all the different individuals
        collated_times: times for all the different timesteps for all the different individuals
    '''
    patient_data_groups = patient_data.groupby(['stay_id'])

    collated_scores = []
    collated_times = []
    patient_scores_obtained = 0
    def test_func(a): return trace_lambda
    for value, (groupStayID, groupData) in enumerate(patient_data_groups):
        if patient_scores_obtained >= num_cases_to_assess:
            break

        hours = groupData[['hours_since_admission']]

        # Normalize the stay by the last value
        normalized_hours = (hours/hours.iloc[-1]).to_numpy()
        groupData.drop(['stay_id', 'hours_since_admission',
                       'RFD'], axis=1, inplace=True)

        factual = groupData.to_numpy()

        # Get the desirable and undesirable counterfactuals for the current time point
        positive_cf_dist, positive_cf_ind = tree_1.query(groupData, k=num_cfs)
        negative_cf_dist, negative_cf_ind = tree_2.query(groupData, k=num_cfs)

        score_values = []
        desirable_cf_scores = []
        undesirable_cf_scores = []
        nrfd_probs = []
        rfd_probs = []
        mortality_probs = []

        times = []

        for i in range(len(factual) - 1):
            # Get the initial point factual and the next point factual
            xt = factual[i, :]
            xt1 = factual[i + 1, :]

            # Calculate the model probability
            # First for the initial timepoint (xt at timepoint 0), later capture it for xt1
            if i == 0:
                model_prediction_probability = model.predict_proba(
                    xt.reshape(1, -1))
                nrfd_probs.append(model_prediction_probability[0][0])
                rfd_probs.append(model_prediction_probability[0][1])
                mortality_probs.append(
                    model_prediction_probability[0][2])

            if sum(xt-xt1) == 0:
                # If there is no change in any feature values between succcesive time points, don't calculate TraCE
                pass

            else:
                # Get the positive cf and the negative cf to calculate the score
                # Checks whether a counterfactual is obtained or not
                if positive_cf_ind[i] is None or negative_cf_ind[i] is None:
                    pass

                else:
                    # Option to use actual patient outcome as positive_cf or not
                    if oracle_desirable_cf:
                        x_prime = factual[-1:][0].astype(float)
                    else:
                        x_prime = ind_rfd_1.iloc[positive_cf_ind[i]].to_numpy()
                        x_prime = x_prime.astype(float)

                    x_star = ind_rfd_2.iloc[negative_cf_ind[i]].to_numpy()
                    x_star = x_star.astype(float)

                    if x_prime.shape[0] > 1:
                        positive_score_value = 0
                        np.argwhere(xt - x_star != 0)
                        num_pos = 0
                        for k in range(x_prime.shape[0]):
                            if oracle_desirable_cf:
                                temp_ind = np.argwhere(xt - x_prime != 0)
                            else:
                                temp_ind = np.argwhere(xt - x_prime[k, :] != 0)
                            temp_xt = xt[temp_ind].reshape((len(temp_ind)))
                            temp_xt1 = xt1[temp_ind].reshape((len(temp_ind)))
                            if oracle_desirable_cf:
                                temp_xprime = x_prime[temp_ind].reshape(
                                    (len(temp_ind)))
                            else:
                                temp_xprime = x_prime[k, temp_ind].reshape(
                                    (len(temp_ind)))
                            temp_bug1 = np.sum(temp_xt - temp_xt1)
                            if temp_bug1 != 0:
                                num_pos += 1
                                positive_score_value += score(
                                    temp_xt, temp_xt1, temp_xprime, func=test_func)

                        negative_score_value = 0
                        num_neg = 0
                        for k in range(x_star.shape[0]):
                            temp_ind = np.argwhere(xt - x_star[k, :] != 0)
                            temp_xt = xt[temp_ind].reshape((len(temp_ind)))
                            temp_xt1 = xt1[temp_ind].reshape((len(temp_ind)))
                            temp_x_star = x_star[k, temp_ind].reshape(
                                (len(temp_ind)))
                            temp_bug1 = np.sum(temp_xt - temp_xt1)
                            if temp_bug1 != 0:
                                num_neg += 1
                                negative_score_value += score(
                                    temp_xt, temp_xt1, temp_x_star, func=test_func)
                        if num_pos == 0 or num_neg == 0:
                            positive_score_value = 0
                            negative_score_value = 0
                        else:
                            positive_score_value = positive_score_value / num_pos
                            negative_score_value = negative_score_value / num_neg

                    else:
                        # Desirable CFE component
                        positive_score_value = score(
                            xt, xt1, x_prime, func=test_func)

                        # Undesirable CFE component
                        negative_score_value = score(
                            xt, xt1, x_star, func=test_func)

                    score_value = (positive_score_value -
                                   negative_score_value) / 2
                    score_values.append(score_value)
                    desirable_cf_scores.append(positive_score_value)
                    undesirable_cf_scores.append(negative_score_value)
                    times.append(normalized_hours[i])

                    # Also calculate the model probability
                    model_prediction_probability = model.predict_proba(
                        xt1.reshape(1, -1))
                    nrfd_probs.append(model_prediction_probability[0][0])
                    rfd_probs.append(model_prediction_probability[0][1])
                    mortality_probs.append(model_prediction_probability[0][2])

        if len(score_values) > 0:

            collated_scores.append(score_values)
            collated_times.append(times)

            # plot TraCE scores over the stay
            # offset x range to begin at 1
            # to reflect TraCE method (measure timepoint and preceeding one)
            x_range = range(1, len(desirable_cf_scores) + 1)
            plt.style.use('seaborn-paper')
            plt.rcParams["axes.spines.right"] = False
            plt.rcParams["axes.spines.top"] = False
            plt.plot(x_range, desirable_cf_scores,
                     label=f'Desirable: {np.mean(desirable_cf_scores):.2f}')
            plt.plot(x_range, undesirable_cf_scores,
                     label=f'Undesirable: {np.mean(undesirable_cf_scores):.2f} ')
            plt.plot(x_range, score_values,
                     label=f'Total TraCE: {np.mean(score_values):.2f}')
            # plt.xticks(x_range)
            # Add horizontal dashed line at 0
            plt.axhline(0, linestyle='--', c='black')
            plt.xlabel('ICU Stay Timepoint')
            plt.ylabel('TraCE Score')
            plt.xlim(0, len(desirable_cf_scores)+1)
            plt.ylim(-1, 1)
            plt.tight_layout()
            plt.legend()
            if oracle_desirable_cf:
                plt.savefig(
                    f'plots/plot_patient_{patient_scores_obtained}_{label}_TraCE_KDTree_oracle.pdf')
            else:
                plt.savefig(
                    f'plots/plot_patient_{patient_scores_obtained}_{label}_TraCE_KDTree.pdf')
            plt.show(block=False)
            plt.close('all')

            # plot probabilities
            plt.plot(nrfd_probs, label='NRFD', color='m')
            plt.plot(rfd_probs, label='RFD', color='blue')
            plt.plot(mortality_probs, label='Mortality', color='orange')
            plt.xlabel('ICU Stay Timepoint')
            plt.ylabel('Probability')
            plt.xlim(0, len(nrfd_probs))
            plt.ylim(0, 1)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f'plots/plot_patient_{patient_scores_obtained}_{label}_probs.pdf')
            plt.show(block=False)
            plt.close('all')

            patient_scores_obtained += 1
            print(
                f'Patient scores obtained: {patient_scores_obtained} / {num_cases_to_assess}')

    return collated_scores, collated_times


def analysis(filename,
             cumulative=False
             ):
    '''
    Function used to calculate the:
    mean, sum and standard deviation of the scores (in total as well as per patient)

    inputs:
        filename - name of the pickle file which contains the data
        cumulative - whether to calculate the cumulative average or not

    returns: full_sum - sum of all the traCE scores
             full_mean - mean of all the traCE scores
             full_std - std of all the traCE scores
             patient_sums - sum of all the traCE scores for each individual patient
             patient_means - mean of all the traCE scores
             patient_std - std of all the traCE scores
    '''

    with open(filename, 'rb') as file:
        # Use pickle.load() to load the object from the file
        scores = pickle.load(file)

    # Calculate the mean for each sublist
    patient_sums = []
    patient_means = []
    patient_std = []
    cum_sum_list = []
    final_timepoints = []
    for sublist in scores:
        if cumulative:
            sublist = cumulative_average_trace(sublist)
            cum_sum_list.append(sublist)
        total = np.sum(sublist)
        patient_sums.append(total)

        mean = np.mean(sublist)
        patient_means.append(mean)

        std_val = np.std(sublist)
        patient_std.append(std_val)
        final_timepoints.append(sublist[-1:])

    if cumulative:
        full_std = np.std(np.concatenate(cum_sum_list))
        full_mean = np.mean(np.concatenate(cum_sum_list))
        return full_mean, full_std
    else:
        full_mean_score = np.mean(patient_means)
        full_std_score = np.std(patient_means)
        return full_mean_score, full_std_score, patient_means


if __name__ == "__main__":
    path = 'full_datatable_timeSeries_Labels.csv'
    positive_patient_info, negative_patient_info = generate_dice_cf_global(
        path,
        trace_lambda=0.9,
        num_cases_to_assess=10,
        num_cfs=2,
        oracle_desirable_cf=False)
    print('-------------- Processing complete, summary: --------------')
    print('--- Pos outcomes --- mean, std', positive_patient_info)
    print('--- Neg outcomes --- mean, std', negative_patient_info)
