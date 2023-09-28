import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# for some reason I cannot install geopandas :( this function is also in funcs_ssp_study, so feel free to swap
def get_country(path, country_code, frequency, feature, date_index):
    """
    Fetches and normalises a country's ssp and historical data for a given feature in a given time frame
    Args:
        path: path to data files
        country_code: code of the country for analysis
        frequency: daily, monthly, etc
        feature: fetched feature
        date_index: dates

    Returns:

    """
    features = []
    csv_files = [path + "historical_" + feature + ".csv",
                 path + "ssp1_" + feature + ".csv",
                 path + "ssp2_" + feature + ".csv",
                 path + "ssp3_" + feature + ".csv",
                 path + "ssp4_" + feature + ".csv",
                 path + "ssp5_" + feature + ".csv"]
    for file in csv_files:
        # Read the csv file
        df = pd.read_csv(file)

        # Convert the date column to datetime format
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])

        # Set the datetime index and remove the name
        df.set_index(df.columns[0], inplace=True)
        df.index.name = None

        # Trim the dataframe to the date range
        df = df.loc[date_index]

        # Aggregate to annual time steps
        df = df.resample(frequency).mean()

        # Get the features
        data = df[country_code].values

        # Append to the features list
        features.append(data)

    # normalise the whole data frame
    min_val = np.min(features)
    max_val = np.max(features)

    # Scale the array to be between 0 and 1
    output_data = (features - min_val) / (max_val - min_val)

    return output_data

path = "data/ssp_data/"
start_date = "2015-02-01"
end_date = "2021-12-31"
frequency = "M"
country_code = "USA"
features = ["ch4", "gdp", "population", "precipitation", "temperature"]
dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
totals = np.zeros((len(features), 5, len(dates)))
fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(12, 8))
row = 0
col = 0
for count1, feature in enumerate(features):
    datas = get_country(path, country_code, frequency, feature, dates)
    max_y = 0
    for count2, data in enumerate(datas[1::]):
        temp = np.sqrt(np.cumsum((datas[0] - data)**2))
        if np.max(temp) > max_y:
            max_y = np.max(temp)
        ax[row, col].plot(dates, temp, label='ssp ' + str(count2 + 1))
        ax[row, col].set_xlim([np.min(dates), np.max(dates)])
        ax[row, col].set_ylim([0, np.max(temp)])
        totals[count1, count2, :] = temp
    ax[row, col].set_title(feature)
    ax[row, col].set_xlim([np.min(dates), np.max(dates)])
    ax[row, col].set_ylim([0, max_y])
    totals[count1, count2, :] = temp
    plt.ylabel('Difference between SSP and Ground Truth')
    plt.xlabel('Time')

    col += 1
    if col > 2:
        row = 1
        col = 0

for i in range(5):
    temp = totals[:, i, :].sum(axis=0)
    ax[1, 2].plot(dates, temp, label='ssp ' + str(i + 1))

plt.legend(loc='upper left')
ax[1, 2].set_title("Total Contribution to SSP")
plt.ylabel('Difference between SSP and Ground Truth')
fig.suptitle(country_code, fontsize="x-large")
plt.savefig("plots/norm/split/" + country_code + ".pdf")

fig = plt.figure(figsize=(6, 3.5))
temp_max = 0
for i in range(5):
    temp = totals[:, i, :].sum(axis=0)
    plt.plot(dates, temp, label='ssp ' + str(i + 1))
    if temp_max < np.max(temp):
        temp_max = np.max(temp)
plt.legend(loc='upper left')
plt.title(country_code + " Total Contribution to SSP")
plt.ylabel('Difference between SSP and Ground Truth')
plt.xlabel('Year')
plt.xlim([np.min(dates), np.max(dates)])
plt.ylim([0, temp_max])
plt.savefig("plots/norm/totals/" + country_code + ".pdf")
