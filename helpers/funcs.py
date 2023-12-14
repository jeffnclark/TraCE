import numpy as np
from scipy.special import gamma, factorial
import pandas as pd


def vec(x1, x2):
    """
    Create the vector that defines the linear movement between two points
    :param x1: start point
    :param x2: end point
    :return: vector between points
    """
    return x2 - x1


def cos_sim(v, u):
    """
    calculate the normalised angle between two vectors.
    u and v must be vectors of size n
    :param v: vector 1
    :param u: vector 2
    :return: similarity [-1, 1]
    """
    v_len = np.linalg.norm(v)
    u_len = np.linalg.norm(u)
    dot = np.dot(v, u)
    return dot / (v_len * u_len)


def n_sphere(v):
    r = np.linalg.norm(v)
    n = len(v)
    V = (np.pi ** (n / 2) / (gamma(n / 2 + 1))) * r ** n
    return V


def score(x0, x1, x_prime, func=None):
    """
    return the score of the trajectory as a balance between where we want to go, where we
    don't want to go, and where we actually went
    :param x0: start point
    :param x1: end point
    :param x_prime: counterfactual
    :return: score
    """
    v = vec(x0, x1)
    v_prime0 = vec(x0, x_prime)
    v_prime1 = vec(x1, x_prime)
    length = np.linalg.norm

    phi = np.dot(v_prime0, v) / (length(v_prime0) * length(v))

    if phi == 1:
        psi = length(v) / length(v_prime0)
    elif phi <= 0:
        v_best = v_prime0
        psi = np.dot(v_best, v_prime1) / (length(v_best) * length(v_prime1))
    else:
        mag = length(v_prime0) * phi
        x_best = x0 + (v / length(v)) * mag
        v_best = x_prime - x_best
        psi = np.dot(v_best, v_prime1) / (length(v_best) * length(v_prime1))

    if func is None:
        lam = np.exp(-(length(v_prime1 - v_prime0)))
    else:
        lam = func(length(v_prime1))

    S = lam * phi + (1 - lam) * psi

    return S


def score2(x0, x1, x_prime, func=None, weight=0.5):
    """
    return the score of the trajectory as a balance between where we want to go, where we
    don't want to go, and where we actually went
    :param x0: start point
    :param x1: end point
    :param x_prime: counterfactual
    :return: score
    """
    v = vec(x0, x1)
    v_prime0 = vec(x0, x_prime)
    v_prime1 = vec(x1, x_prime)
    length = np.linalg.norm

    if func is None:
        h = 2 / (1 + np.exp(-(n_sphere(v_prime0) - n_sphere(v_prime1)))) - 1
    else:
        h = func(x0, x1, x_prime)

    f = 1 / 2 * (np.dot(v, v_prime0) / (length(v) * length(v_prime0)) +
                 np.dot(v, v_prime1) / (length(v) * length(v_prime1)))
    if f == 0:
        S = h
    else:
        S = weight * f + (1 - weight) * h

    return S


def score1(x0, x1, x_prime, x_star=None, method='dot'):
    """
    return the score of the trajectory as a balance between where we want to go, where we
    don't want to go, and where we actually went
    :param method: method for scoring (dot, avg, interp)
    :param x0: start point
    :param x1: end point
    :param x_prime: positive counterfactual
    :param x_star: negative counterfactual
    :return: score
    """
    if method == 'dot':
        if x_star is None:
            v = vec(x0, x1)
            v_prime = vec(x0, x_prime)
            S = cos_sim(v, v_prime)

        else:
            # get vectors
            v = vec(x0, x1)
            v_prime = vec(x0, x_prime)
            v_star = vec(x0, x_star)

            # get angle
            theta = cos_sim(v_prime, v_star)

            if theta == 1:
                print('Counterfactuals aligned, cannot calculate score')
                return np.nan

            # scale score by maximum possible potential score
            norm_v_prime = v_prime / np.linalg.norm(v_prime)
            norm_v_star = v_star / np.linalg.norm(v_star)
            norm_v = v / np.linalg.norm(v)
            length = np.sqrt(2 - 2 * theta)
            S = np.dot((norm_v_prime - norm_v_star) / length, norm_v)

    if method == 'avg':
        S = 0
        v = vec(x0, x1)
        norm_v = v / np.linalg.norm(v)
        n = 0
        for xp in x_prime:
            v_prime = vec(x0, xp)
            p_test1 = (np.dot(v, v_prime) /
                       (np.linalg.norm(v_prime) * np.linalg.norm(v)))
            p_test2 = (1 - np.exp(-np.linalg.norm(v)))
            p_test3 = (np.exp(-np.linalg.norm(x1 - xp)))
            temp = (np.dot(v, v_prime) / (np.linalg.norm(v_prime) * np.linalg.norm(v))) * (
                1 - np.exp(-np.linalg.norm(v))) * (np.exp(-np.linalg.norm(x1 - xp)))
            S += temp
            n += 1
        for xs in x_star:
            v_star = vec(x0, xs)
            s_test1 = (np.dot(v, v_star) /
                       (np.linalg.norm(v_star) * np.linalg.norm(v)))
            s_test2 = (1 - np.exp(-np.linalg.norm(v)))
            s_test3 = (np.exp(-np.linalg.norm(x1 - xs)))
            temp = (np.dot(v, v_star) / (np.linalg.norm(v_star) * np.linalg.norm(v))) * (
                1 - np.exp(-np.linalg.norm(v))) * (np.exp(-np.linalg.norm(x1 - xs)))
            S -= temp
            n += 1

        S = S / n

    if method == 'interp':
        S = 0
        # get vectors as normalised differences
        v = vec(x0, x1)
        v_prime = np.zeros(x_prime.shape)
        for i, prime in enumerate(x_prime):
            temp = vec(x0, prime)
            v_prime[i] = temp / np.linalg.norm(temp)
        v_star = np.zeros(x_star.shape)
        for i, star in enumerate(x_star):
            temp = vec(x0, star)
            v_star[i] = temp / np.linalg.norm(temp)

        # convert to polar

    return S


def news(x):
    """
    Calculates the NEWS score, whilst capturing the good counterfactual and bad counterfactual in parallel
    :param x: vector of NEWS values for all features
    :return: news score, good counterfactual x_prime, bad counterfactual x_star
    """
    # set up values for calculation
    score = 0
    x_prime = np.zeros(len(x))
    x_star = np.zeros(len(x))
    # perfect values from mean of 0 score
    x_prime[0] = 16  # respiration rate
    x_prime[1] = 100  # oxygen saturation levels
    x_prime[2] = 0  # any supplemental oxygen
    x_prime[3] = 37.05  # temperature
    x_prime[4] = 165  # blood pressure
    x_prime[5] = 70.5  # heart rate
    x_prime[6] = 0  # consciousness

    # calculate respiratory score
    if x[0] <= x_prime[0]:
        x_star[0] = 12
    else:
        x_star[0] = 20
    if x[0] <= 11:
        score += 1
        x_star[0] = 8
    if x[0] <= 8:
        score += 2
        x_star[0] = x[0] - 1
    if x[0] >= 21:
        score += 2
        x_star[0] = 25
    if x[0] >= 25:
        score += 1
        x_star[0] = x[0]

    # calculate oxygen saturation score
    if x[1] >= 96:
        x_star[1] = 95
    if x[1] <= 95:
        score += 1
        x_star[1] = 93
    if x[1] <= 93:
        score += 1
        x_star[1] = 91
    if x[1] <= 91:
        score += 1
        x_star[1] = x[1]

    # calculate supplemental oxygen score (1 on supplementary oxygen, 0 not)
    x_star[2] = 1
    if x[2] == 1:
        score += 2

    # calculate temperature score
    if x[3] <= x_prime[3]:
        x_star[3] = 36
    else:
        x_star[3] = 38.1
    if x[3] <= 36:
        score += 1
        x_star[3] = 35
    if x[3] <= 35:
        score += 2
        x_star[3] = x[3] - 1
    if x[3] >= 38.1:
        score += 1
        x_star[3] = 39.1
    if x[3] >= 39.1:
        score += 1
        x_star[3] = x[3]

    # calculate blood pressure score
    if x[4] <= x_prime[4]:
        x_star[4] = 110
    else:
        x_star[4] = 220
    if x[4] <= 110:
        score += 1
        x_star[4] = 100
    if x[4] <= 100:
        score += 1
        x_star[4] = 90
    if x[4] <= 90:
        score += 1
        x_star[4] = x[4]
    if x[4] >= 220:
        score += 3
        x_star[4] = x[4]

    # calculate heart rate score
    if x[5] <= x_prime[5]:
        x_star[5] = 50
    else:
        x_star[5] = 91
    if x[5] <= 50:
        score += 1
        x_star[5] = 40
    if x[5] <= 40:
        score += 1
        x_star[5] = x[5]
    if x[5] >= 91:
        score += 1
        x_star[5] = 111
    if x[5] >= 111:
        score += 1
        x_star[5] = 131
    if x[5] >= 131:
        score += 1
        x_star[5] = x[0]

    # calculate level of consciousness score (1 unconscious, 0 conscious)
    x_star[6] = 1
    if x[6] == 1:
        score += 3

    return score, x_prime, x_star


def cumulative_average_trace(scores):
    '''
    Calculate cumulative average TraCE score
    The final value is an average of the entire trajectory

    '''
    df = pd.DataFrame(scores, columns=['score'])
    return df['score'].expanding().mean().to_numpy()


def normalize_data(df_data):
    '''
    function used to normalize the dataframe
    input: df_data -  pd dataframe to be normalized
    returns : normalized_df_data - dataframe after normalization
    '''
    # Calculate separate mean and standard deviation values
    df_mean = df_data.mean()
    df_std = df_data.std()
    normalized_df_data = (df_data-df_mean)/df_std
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
                         'fio2', 'haemoglobin', 'heartRate', 'motorGCS', 'eyeGCS',
                         'potassium', 'respiratoryRate', 'sodium', 'Temperature [C]',
                         'verbalGCS', 'age', 'gender',
                         'hours_since_admission',
                         'RFD']

    # Obtain only the variables of interes
    desired_df_data = df_data[desired_variables]
    # Make sure to use desired_df_columnss for the len threshold
    filtered_df_data = desired_df_data.dropna(
        thresh=len(desired_df_data.columns))
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
    # Separate the data into labels, an example here is the ready for discharge ICU case study
    neutral_data = data[data_labels['RFD'] == 0]
    positive_data = data[data_labels['RFD'] == 1]
    negative_data = data[data_labels['RFD'] == 2]

    # Get the minimum values
    values = [len(neutral_data), len(negative_data), len(positive_data)]
    min_value = min(values)
    # Shorten the array based on the labels
    neutral_data = neutral_data.head(min_value)
    negative_data = negative_data.head(min_value)
    positive_data = positive_data.head(min_value)

    # Shorten the labels
    neutral_labels = data_labels[data_labels['RFD'] == 0].head(min_value)
    positive_labels = data_labels[data_labels['RFD'] == 1].head(min_value)
    negative_labels = data_labels[data_labels['RFD'] == 2].head(min_value)

    # Concatenate the data
    balanced_df_data = pd.concat(
        (neutral_data, negative_data, positive_data), axis=0)
    balanced_df_labels = pd.concat(
        (neutral_labels, negative_labels, positive_labels), axis=0)

    return balanced_df_data, balanced_df_labels


def nan_post_processing_data(df_data, columns):
    '''
    Function for post processing the dataframe with all the proceessed values (eg removing nans)
    input: 
    df_data -  dataframe which contain the values of all the processed files
    columns -  column of interest to perform data processing on
    returns:
    df_data -  post-processed dataframe
    '''
    postprocessSettings = 2

    # for column in df_data:
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


def change_categorical(df, categorical_features):
    '''
    function: Used to change data variables to categorical type (for the case of passing to DiCE)
    '''
    for column in categorical_features:
        df[column] = df[column].astype('category')
    return df


def cf_generator(df, classifier):
    ''' Function to make the object for the generation of the counterfactuals
    input: df: dataframe containing all the data, also includes the labels
    classifier: trained deep learnng classifier
    retirms: exp_genetic_mimic - object to generate DiCE counterfactuals
            features_to_vary - features which I want to vary when generating counterfactuals
    '''

    continuous_features_mimic = df.drop(
        columns=['RFD', 'eyeGCS', 'motorGCS', 'verbalGCS', 'gender'], axis=1).columns.tolist()

    # Feature to vary during counterfactual generation
    features_to_vary = df.drop(columns=['RFD', 'age'], axis=1).columns.tolist()
    d_mimic = dice_ml.Data(dataframe=df,
                           continuous_features=continuous_features_mimic,
                           outcome_name='RFD')

    # We provide the type of model as a parameter (model_type)
    m_mimic = dice_ml.Model(
        model=classifier, backend="sklearn", model_type='classifier')
    exp_genetic_mimic = Dice(d_mimic, m_mimic)  # , method="genetic")
    return exp_genetic_mimic, features_to_vary


def initial_icu_processing(filepath, features):
    '''
    Function used to perform initial preprocessing of the data, this includes:
    choosing the correct columns of interest,
    changing the label to be negative outocome label from -1 to 2 (for the purpose of DiCE),
    changing the gender label,
    as well as filling in missing values in the data
    input:
        filepath - path of the csv file which cotains the data
        features - which features you want to extract and use

    return:desired_df_data -  dataframe after processing the data 
    '''
    df_data = pd.read_csv(filepath,
                          header=0)
    # Get the variables of interest

    # Obtain only the variables of interes
    desired_df_data = df_data[features]
    # Replace values of -1 with 2 (NEGATIVE LABELLED CLASSES DOES NOT SEEM TO WORK WITH DICE)
    desired_df_data['RFD'].replace(-1, 2, inplace=True)

    # Change gender category from string to float
    desired_df_data['gender'].replace('M', 0, inplace=True)
    desired_df_data['gender'].replace('F', 1, inplace=True)
    desired_df_data['gender'] = desired_df_data['gender'].astype('float')

    # Update the mean and the standard deviation for the data
    columns_to_process = desired_df_data.drop(
        columns=['RFD', 'stay_id', 'hours_since_admission'], axis=1).columns.tolist()
    # columns_to_process=desired_df_data.columns.tolist()

    neutral_data = desired_df_data[desired_df_data['RFD'] == 0]
    positive_data = desired_df_data[desired_df_data['RFD'] == 1]
    negative_data = desired_df_data[desired_df_data['RFD'] == 2]

    # Fill in missing values of the data
    processed_neutral_data = nan_post_processing_data(
        neutral_data, columns_to_process)
    processed_negative_data = nan_post_processing_data(
        negative_data, columns_to_process)
    processed_positive_data = nan_post_processing_data(
        positive_data, columns_to_process)

    desired_df_data.update(processed_neutral_data)
    desired_df_data.update(processed_negative_data)
    desired_df_data.update(processed_positive_data)

    return desired_df_data
