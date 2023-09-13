from operator import index
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import glob
import sys
import numpy as np
import pathlib
import math
import random
import scipy
from helpers.funcs import *
from helpers.plotters import *
from sklearn.neural_network import MLPClassifier
import dice_ml
from dice_ml import Dice
from sklearn.metrics import f1_score, accuracy_score
import pickle
import warnings

warnings.filterwarnings('ignore')

seed = 42
prob_thresh = 0.67

def normalize_data(df_data):
    '''
    function used to normalize the dataframe
    input: df_data -  pd dataframe to be normalized
    returns : normalized_df_data - dataframe after normalization
    '''
    # Calculate separate mean and standard deviation values
    df_mean = df_data.mean()
    df_std = df_data.std()
    normalized_df_data=(df_data-df_mean)/df_std
    return normalized_df_data, df_mean, df_std

def remove_empty_rows(filepath):
    '''
    Function to get the data where there is the data with only the empty rows
    CURRENTLY UNUSED
    '''
    df_data = pd.read_csv(filepath, 
                    header=0)
    
    desired_variables = ['stay_id',
       'biocarbonate', 'bloodOxygen', 'bloodPressure', 'bun', 'creatinine',
       'fio2', 'haemoglobin', 'heartRate', 'motorGCS','eyeGCS',
       'potassium', 'respiratoryRate', 'sodium', 'Temperature [C]',
       'verbalGCS','age','gender',  
        'hours_since_admission',
        'RFD']
    
    # Obtain only the variables of interes
    desired_df_data = df_data[desired_variables]
    # Make sure to use desired_df_columnss for the len threshold
    filtered_df_data = desired_df_data.dropna(thresh=len(desired_df_data.columns))
    concatenated = pd.concat([desired_df_data, filtered_df_data])
    different_rows = concatenated.drop_duplicates(keep=False)
    # Reset index of the resulting dataframe
    different_rows.reset_index(drop=True, inplace=True)
    return filtered_df_data, different_rows

def balancing_classes(
        data,
        data_labels):
    '''
    Used to balance the data and the data labels
    input: data - can be the data or the data labels which we want to obtain values of
    data_labels- used to obtain the separate classes of interest
    return: balanced_df_data - data which is balanced in terms of the different classes
            balanced_df_labels - labels which is balanced in terms of the different classes
    '''
    # Separate the data into labels
    neutral_data = data[data_labels['RFD']==0]
    positive_data = data[data_labels['RFD']==1]
    negative_data = data[data_labels['RFD']==2]
    
    # Get the minimum values 
    values = [len(neutral_data), len(negative_data), len(positive_data)]
    min_value = min(values)
    # Shorten the array based on the labels
    neutral_data = neutral_data.head(min_value)
    negative_data = negative_data.head(min_value)
    positive_data = positive_data.head(min_value)  

    # Shorten the labels
    neutral_labels = data_labels[data_labels['RFD']==0].head(min_value)
    positive_labels = data_labels[data_labels['RFD']==1].head(min_value)
    negative_labels = data_labels[data_labels['RFD']==2].head(min_value)
    
    # Concatenate the data     
    balanced_df_data = pd.concat((neutral_data,negative_data, positive_data),axis=0)
    balanced_df_labels = pd.concat((neutral_labels,negative_labels, positive_labels),axis=0)
    
    return balanced_df_data, balanced_df_labels

def nan_post_processing_data(df_data,columns):
    '''
    Function for post processing the dataframe with all the proceessed values (eg removing nans)
    input: 
    df_data -  dataframe which contain the values of all the processed files
    columns -  column of interest to perform data processing on
    returns:
    df_data -  post-processed dataframe
    '''    
    postprocessSettings = 2

    #for column in df_data:
    for column in columns:
        # determine replacement values
        if postprocessSettings == 0:
                # leave empty field empty
                newValue = ''
        elif postprocessSettings == 1:
                # replace empty field by a zero
                newValue = 0
        elif postprocessSettings == 2:
            # replace empty field with
            # the mean value for column containing numerical values,
            # and the most frequent value in columns
            # with non-numerical/categorical values
            if 'GCS' in column:
                newValue = df_data[column].mode()
                # FOR THE CASE OF THE MODE, NEED TO GET THE FIRST INDEX
                df_data[column] = df_data[column].fillna(newValue[0])
            elif 'gender' in column:
                newValue = df_data[column].mode()
                df_data[column] = df_data[column].fillna(newValue[0])
            else: 
                newValue = df_data[column].mean()
                df_data[column] = df_data[column].fillna(newValue)
    return df_data


def change_categorical(df):
    '''
    function: Used to change data variables to categorical type (for the case of passing to DiCE
    '''
    categorical_columns = ['motorGCS','verbalGCS','eyeGCS','gender']
    for column in categorical_columns:
        df[column] = df[column].astype('category')
    return df


def initial_processing(filepath):
    '''
    Function used to perform initial preprocessing of the data, this includes choosing the correct columns of interest,
    changing the label to be negative outocome label from -1 to 2 (for the purpose of DiCE), changing the gender label,
    as well as filling in missing values in the data
    input: filepath - path of the csv file which cotains the data
    return:desired_df_data -  dataframe after processing the data 
    '''
    df_data = pd.read_csv(filepath, 
                    header=0)
    # Get the variables of interest
    
    desired_variables = ['stay_id',
       'biocarbonate', 'bloodOxygen', 'bloodPressure', 'bun', 'creatinine',
       'fio2', 'haemoglobin', 'heartRate', 'motorGCS','eyeGCS',
       'potassium', 'respiratoryRate', 'sodium', 'Temperature [C]',
       'verbalGCS','age','gender',  
        'hours_since_admission',
        'RFD']
    
    # Obtain only the variables of interes
    desired_df_data = df_data[desired_variables]
    # Replace values of -1 with 2 (NEGATIVE LABELLED CLASSES DOES NOT SEEM TO WORK WITH DICE)
    desired_df_data['RFD'].replace(-1, 2, inplace=True)

    # Change gender categgory from string to float
    desired_df_data['gender'].replace('M', 0, inplace=True)
    desired_df_data['gender'].replace('F', 1, inplace=True)
    desired_df_data['gender'] = desired_df_data['gender'].astype('float')

    # Update the mean and the standard deviation for the data
    columns_to_process=desired_df_data.drop(columns=['RFD','stay_id','hours_since_admission'], axis=1).columns.tolist()
    #columns_to_process=desired_df_data.columns.tolist()

    neutral_data = desired_df_data[desired_df_data['RFD']==0]
    positive_data = desired_df_data[desired_df_data['RFD']==1]
    negative_data = desired_df_data[desired_df_data['RFD']==2]

    # Fill in missing values of the data
    processed_neutral_data = nan_post_processing_data(neutral_data,columns_to_process)
    processed_negative_data = nan_post_processing_data(negative_data,columns_to_process)
    processed_positive_data = nan_post_processing_data(positive_data,columns_to_process)
    
    desired_df_data.update(processed_neutral_data)
    desired_df_data.update(processed_negative_data)
    desired_df_data.update(processed_positive_data)

    return desired_df_data


def cf_generator(df,classifier):
    ''' Function to make the object for the generation of the counterfactuals
    input: df: dataframe containing all the data, also includes the labels
    classifier: trained deep learnng classifier
    retirms: exp_genetic_mimic - object to generate diCE counterfactuals
            features_to_vary - features which I want to vary when generating counterfactuals
    '''

    continuous_features_mimic = df.drop(columns=['RFD','eyeGCS','motorGCS','verbalGCS','gender'], axis=1).columns.tolist()
    
    # Feature to vary during counterfactual generation
    features_to_vary=df.drop(columns=['RFD','age'], axis=1).columns.tolist()
    d_mimic = dice_ml.Data(dataframe=df,
                      continuous_features=continuous_features_mimic,
                      outcome_name='RFD')
    
    # We provide the type of model as a parameter (model_type)
    m_mimic = dice_ml.Model(model=classifier, backend="sklearn", model_type='classifier')
    exp_genetic_mimic = Dice(d_mimic, m_mimic, method="genetic")
    return exp_genetic_mimic, features_to_vary
    
def obtain_stay_id_individuals(
        data,
        labels,
        label_value
        ):
    '''
    Function used to obtain patients with a specific outcome of interest
    input - data - patient data
            labels - data labels
            label_value - value to choose specific type of patient of interest
    return:full_test_df_stay_id_data - patients who have the stay id of a particular outcome

    '''
    
    # Filtering the stayID for the data of interest
    positive_stay_id = labels[labels['RFD']==label_value]
    full_test_df_stay_id =pd.concat([data,labels[['RFD','stay_id','hours_since_admission']]],axis=1)
    # Based on chatgpt
    full_test_df_stay_id_data = full_test_df_stay_id[full_test_df_stay_id['stay_id'].isin(positive_stay_id['stay_id'])]
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
    filtered_df_data = initial_processing(filepath)

    train_data, test_data = train_test_split(filtered_df_data, test_size=0.2, random_state=seed, shuffle=False)

    # Remove columns which are important in general but not for the training of the classifier'
    y_train = train_data[['stay_id', 'RFD','hours_since_admission']]
    x_train = train_data.drop(columns=['stay_id', 'RFD','hours_since_admission'])

    y_test = test_data[['stay_id', 'RFD','hours_since_admission']]
    x_test = test_data.drop(columns=['stay_id', 'RFD','hours_since_admission'])
    # Normalize the data
    x_train_norm, df_mean, df_std = normalize_data(x_train)
    x_test_norm = (x_test-df_mean)/df_std
    full_norm = (filtered_df_data-df_mean)/df_std

    # Balance the data points belonging to the different classes in the train and test set
    balanced_x_train_norm, balanced_y_train = balancing_classes(x_train_norm,y_train)
    balanced_x_test_norm,balanced_y_test = balancing_classes(x_test_norm,y_test)
    # Train classifier
    classifier = MLPClassifier(hidden_layer_sizes=(10, 10), random_state=seed, max_iter=10).fit(balanced_x_train_norm, balanced_y_train['RFD']) 
    # Evaluate classifer performance
    train_predictions = classifier.predict(balanced_x_train_norm)
    f1_train_score, accuracy_train_score = f1_score(balanced_y_train['RFD'],train_predictions,average='macro'), accuracy_score(balanced_y_train['RFD'],train_predictions)    
    test_predictions = classifier.predict(balanced_x_test_norm)
    f1_test_score, accuracy_test_score = f1_score(balanced_y_test['RFD'],test_predictions,average='macro'), accuracy_score(balanced_y_test['RFD'],test_predictions)
    
    print('train f1',f1_train_score, 'train accuracy',accuracy_train_score)
    print('f1_test_score',f1_test_score, 'test accuracy',accuracy_test_score)

    # Evaluate classifer performance - original unbalanced test set
    unbalanced_test_predictions = classifier.predict(x_test_norm)
    f1_unbalanced_test_score, accuracy_unbalanced_test_score = f1_score(y_test['RFD'],unbalanced_test_predictions,average='macro'), accuracy_score(y_test['RFD'],unbalanced_test_predictions)
   
    print('f1_unbalanced_test_score',f1_unbalanced_test_score, 'unbalanced_test accuracy',accuracy_unbalanced_test_score)

    return classifier, balanced_x_train_norm,balanced_y_train,x_test_norm,y_test


def generate_dice_cf_global(filepath,
                            num_cases_to_assess = 3,
                            num_cfs = 1,
                            num_trace_stays=5,
                            oracle_desirable_cf = False,
                            ):
    ''' Generate plot for a subset of the individuals'''
    trained_model,processed_balanced_x_train_norm,balanced_y_train,processed_x_test_norm,y_test = train_classifier(filepath)
    # Join the labels to the data
    full_df = pd.concat([processed_balanced_x_train_norm, balanced_y_train['RFD']],axis=1)
    full_test_df =pd.concat([processed_x_test_norm,y_test['RFD']],axis=1) 
    full_df =  pd.concat([full_df,full_test_df],axis=0)
    
    exp_genetic_mimic,features_to_vary = cf_generator(full_df,trained_model)

    # Includes the patient and stay ID of an individual
    positive_stay_id_patients = obtain_stay_id_individuals(processed_x_test_norm,y_test,1)
    negative_stay_id_patients = obtain_stay_id_individuals(processed_x_test_norm,y_test,2)

    print('------------ INITIATING FOR POSITIVE OUTCOME(S) - RFD -------------')
    positive_patient_scores, positive_times = calculate_traCE_scores(exp_genetic_mimic,positive_stay_id_patients,features_to_vary, num_cases_to_assess=num_cases_to_assess, num_cfs=num_cfs, oracle_desirable_cf=oracle_desirable_cf, label = 'pos')
    print('------------ INITIATING FOR NEGATIVE OUTCOME(S) - MORTALITY -------------')
    negative_patient_scores, negative_times = calculate_traCE_scores(exp_genetic_mimic,negative_stay_id_patients,features_to_vary, num_cases_to_assess=num_cases_to_assess, num_cfs=num_cfs, oracle_desirable_cf=oracle_desirable_cf, label = 'neg')
    
    # Save the pickle files
    positive_patient_score_file_name = 'positive_patient_scores.pkl'
    with open(positive_patient_score_file_name, 'wb') as file:
        pickle.dump(positive_patient_scores, file)
    
    negative_patient_score_file_name = 'negative_patient_scores.pkl'
    with open(negative_patient_score_file_name, 'wb') as file:
        pickle.dump(negative_patient_scores, file)

    # Obtains the values of interest for processing
    positive_full_sum, posiive_full_mean, positive_full_std, positive_patient_sums, positive_patient_means, positive_patient_std = analysis(positive_patient_score_file_name,cumulative=True)
    negative_full_sum, negative_full_mean, negative_full_std, negative_patient_sums, negative_patient_means, negative_patient_std = analysis(negative_patient_score_file_name,cumulative=True)
    return (positive_full_sum, posiive_full_mean, positive_full_std, positive_patient_sums, positive_patient_means, positive_patient_std), (negative_full_sum, negative_full_mean, negative_full_std, negative_patient_sums, negative_patient_means, negative_patient_std)
    

def calculate_traCE_scores(
        dice_generator,
        patient_data,
        features_to_vary,
        num_cases_to_assess=3,
        num_cfs=1,
        oracle_desirable_cf = False,
        label=''
        ):
    '''
    Function used to generate the traCE scores
    inputs: dice_generator: object used to generate the counterfactuals
            patient_data: dataframe of the patients (usually linked to certain class e.g all patients raedy to be discharged)
            features_to_vary: features to vary to generate the counterfactuals using diCE

            oracle_desirable_cf: utilises case's actual final timepoint as the desired counterfactual
    return: 
            collated_scores: scores for all the different timesteps for all the different individuals
            collated_times: times for all the different timesteps for all the different individuals
    '''
    patient_data_groups = patient_data.groupby(['stay_id'])
    collated_scores = []
    collated_times = []
    patient_scores_obtained = 0
    test_func = lambda a : 0.9
    for value, (groupStayID, groupData) in enumerate(patient_data_groups):
        print('patient scores obtained', patient_scores_obtained)
        if patient_scores_obtained >= num_cases_to_assess:
            break
        # Remove stayID
        
        hours = groupData[['hours_since_admission']]

        # Normalize the stay by the last value
        normalized_hours = (hours/hours.iloc[-1]).to_numpy()
        groupData.drop(['stay_id','hours_since_admission','RFD'],axis=1,inplace=True)
        
        # Get the counterfactuals for the positive and the negative data
        # Change to numpy just to make it easier to use
        positive_cf = dice_generator.generate_counterfactuals(groupData, total_CFs=num_cfs, desired_class=1,features_to_vary=features_to_vary,stopping_threshold=prob_thresh)
        negative_cf = dice_generator.generate_counterfactuals(groupData, total_CFs=num_cfs, desired_class=2,features_to_vary=features_to_vary,stopping_threshold=prob_thresh)
        trajectory = groupData.to_numpy()

        # Get all data points except last one (as that would be a discharge or death case)
        factual = trajectory[:-1]
        #vals = 0
        score_values = []
        desirable_cf_scores = []
        undesirable_cf_scores = []

        times = []
        for i in range(len(factual) - 1):
        # Nawid- get the initial point factual and the next point factual
            xt = factual[i, :]
            xt1 = factual[i + 1, :]
            if sum(xt-xt1) ==0:
                 pass
            else:
            # Nawid - get the positive cf and the negative cf to calculate the score
                # Checks whether I obtain a counterfactual or not
                if positive_cf.cf_examples_list[i].final_cfs_df is None or negative_cf.cf_examples_list[i].final_cfs_df is None: 
                    pass
                else:
                    # Option to use actual patient outcome as positive_cf or not
                    if oracle_desirable_cf:
                        x_prime = trajectory[-1:][0].astype(float)
                    else:
                        x_prime = positive_cf.cf_examples_list[i].final_cfs_df.to_numpy()[:,:-1].flatten()
                        x_prime = x_prime.astype(float)

                    x_star = negative_cf.cf_examples_list[i].final_cfs_df.to_numpy()[:,:-1].flatten()
                    x_star = x_star.astype(float)
                
                    positive_score_value = score(xt, xt1, x_prime,func=test_func)
                    negative_score_value = score(xt, xt1, x_star,func=test_func)

                    print('Desirable CF component:', positive_score_value)
                    print('Undesirable CF component:', negative_score_value)

                    score_value = (positive_score_value - negative_score_value)/2
                    score_values.append(score_value)
                    desirable_cf_scores.append(positive_score_value)
                    undesirable_cf_scores.append(negative_score_value)
                    times.append(normalized_hours[i])
    
        if len(score_values)>0:

            collated_scores.append(score_values)
            collated_times.append(times)
            patient_scores_obtained +=1
            print(f'Scores for patient {patient_scores_obtained}: {score_values}')

            plt.plot(desirable_cf_scores, label='Desirable')
            plt.plot(undesirable_cf_scores, label = 'Undesirable')
            plt.plot(score_values, label='Total TraCE')
            plt.legend()
            if oracle_desirable_cf:
                plt.title(f'Test set case {patient_scores_obtained} oracle case')
                plt.savefig(f'plot_patient_{patient_scores_obtained}_{label}_oracle.pdf')
            else:
                plt.title(f'Test set case {patient_scores_obtained}')
                plt.savefig(f'plot_patient_{patient_scores_obtained}_{label}.pdf')
            plt.show(block=False)
            plt.close('all')

    return collated_scores, collated_times

def analysis(filename,cumulative=False):
    '''
    Function used to calculate the mean, sum and standard deviation of the scores (in total as well as per patient)
    
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
    cum_sum_list =[]
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
    
    if cumulative:
        full_sum = np.sum(np.concatenate(cum_sum_list))
        full_std = np.std(np.concatenate(cum_sum_list))
        full_mean = np.mean(np.concatenate(cum_sum_list))
    else:
        full_sum = np.sum(np.concatenate(scores))
        full_std = np.std(np.concatenate(scores))
        full_mean = np.mean(np.concatenate(scores))
    return full_sum, full_mean, full_std, patient_sums, patient_means, patient_std


if __name__ == "__main__":
    path = 'full_datatable_timeSeries_Labels.csv'
    positive_patient_info, negative_patient_info = generate_dice_cf_global(
        path,
        num_cases_to_assess=20,
        num_cfs=1,
        oracle_desirable_cf=False)
    print('--- Pos outcomes ---', positive_patient_info)
    print('--- Neg outcomes ---', negative_patient_info)