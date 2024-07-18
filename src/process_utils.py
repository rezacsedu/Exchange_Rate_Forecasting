# Function to convert dates with mixed formats
def convert_dates(date_str):
  try:
    # First, try parsing the 'YYYYMMDD' format
    return pd.to_datetime(date_str, format='%Y%m%d', errors='coerce')
  except ValueError:
    # If there's a ValueError, try parsing the 'M/DD/YYYY' format
    return pd.to_datetime(date_str, format='%m/%d/%Y', errors='coerce')

# Function to convert dates with mixed formats
def convert_dates_for_aux(date_str):
  # If there's a ValueError, try parsing the 'M/DD/YYYY' format
  return pd.to_datetime(date_str, format='%m/%d/%Y', errors='coerce')
  
def process_main_indexes(INDEX_MACRO_FILE_PATH, recent_data_only):
  # Assuming EUR_INDEX_MACRO_FILE_PATH is defined and points to your CSV file
  df = pd.read_csv(INDEX_MACRO_FILE_PATH, header=[0, 1], dtype={'ECO_RELEASE_DT': str})

  # Create a list of tuples containing the index name and its corresponding columns
  index_columns = [(col[0], col[1]) for col in df.columns]

  # Prepare a list to hold the processed data
  processed_data = []
  num_index_col = 3

  # Iterate over the rows in the dataframe
  for _, row in df.iterrows():
    # Iterate over the index_columns list in steps of 3
    for i in range(0, len(index_columns), num_index_col):
      # Extract the index name and the corresponding ECO_RELEASE_DT, ACTUAL_RELEASE, BN_SURVEY_MEDIAN
      index_name, _ = index_columns[i]
      eco_release_dt = row[index_columns[i]]
      actual_release = row[index_columns[i+1]]
      bn_survey_median = row[index_columns[i+2]]

      # Append the data to the processed_data list
      processed_data.append({
      'Index': index_name,
      'ECO_RELEASE_DT': eco_release_dt,
      'ACTUAL_RELEASE': actual_release,
      'BN_SURVEY_MEDIAN': bn_survey_median
      })

  # Create a new dataframe from the processed data
  df = pd.DataFrame(processed_data)

  # Apply the function to the ECO_RELEASE_DT column
  df['ECO_RELEASE_DT'] = df['ECO_RELEASE_DT'].apply(convert_dates)

  if recent_data_only:
      df = df[(df['ECO_RELEASE_DT'] > '2010-01-01') & (df['ECO_RELEASE_DT'] < '2024-04-01')]

  return df
  
def process_aux_indexes(INDEX_MACRO_FILE_PATH, recent_data_only):
  # Assuming EUR_INDEX_MACRO_FILE_PATH is defined and points to your CSV file
  df = pd.read_csv(INDEX_MACRO_FILE_PATH)

  # Apply the function to the ECO_RELEASE_DT column
  df['Date'] = df['Date'].apply(convert_dates_for_aux)

  if recent_data_only:
    df = df[(df['Date'] > '2010-01-01') & (df['Date'] < '2024-04-01')]

  return df
  
def calculate_surprise_factor(df, recent_data_only):
  # Calculate the surprise factor as the difference between release value and market expectation
  df['Surprise'] = df['ACTUAL_RELEASE'] - df['BN_SURVEY_MEDIAN']

  # Feature engineering: calculate surprises as a percentage of the median
  df['Surprise_pct'] = df['Surprise'] / df['BN_SURVEY_MEDIAN']

  if recent_data_only:
    df = df[(df['ECO_RELEASE_DT'] > '2010-01-01') & (df['ECO_RELEASE_DT'] < '2024-04-01')]

  # # There are NULLS in the data. That's why I considered more recent data, e.g., 2010 to 2024.
  # df = df[(eur_macro_df['ECO_RELEASE_DT'] > '2010-01-01') & (df['ECO_RELEASE_DT'] < '2024-04-01')]

  return df[['Index', 'ECO_RELEASE_DT', 'ACTUAL_RELEASE', 'BN_SURVEY_MEDIAN', 'Surprise', 'Surprise_pct']]

def process_fx_rate_data(INDEX_MACRO_FILE_PATH, recent_data_only):
  # Assuming EUR_INDEX_MACRO_FILE_PATH is defined and points to your CSV file
  df = pd.read_csv(INDEX_MACRO_FILE_PATH)

  # Assuming 'df' is your DataFrame, define a mapping of old column names to new column names with a slash
  column_mapping = {
  'GBPUSD': 'GBP/USD',
  'EURUSD': 'EUR/USD',
  'USDCHF': 'USD/CHF',
  'USDGBP': 'USD/GBP',
  'USDEUR': 'USD/EUR',
  'CHFUSD': 'CHF/USD'
  }

  # Rename the columns using the mapping
  df.rename(columns=column_mapping, inplace=True)

  # Assuming 'df' is your DataFrame, strip leading/trailing spaces and replace multiple spaces with a single space
  df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

  # Apply the function to the ECO_RELEASE_DT column
  df['Date'] = df['Date'].apply(convert_dates_for_aux)

  if recent_data_only:
    df = df[(df['Date'] > '2010-01-01') & (df['Date'] < '2024-04-01')]

  return df
  
def transform_economic_indices(df):
  # Convert 'ECO_RELEASE_DT' to datetime
  df['ECO_RELEASE_DT'] = pd.to_datetime(df['ECO_RELEASE_DT'])

  # Aggregate the data to remove duplicates
  aggregated_df = df.groupby(['Index', 'ECO_RELEASE_DT']).agg({
  'ACTUAL_RELEASE': 'mean',
  'BN_SURVEY_MEDIAN': 'mean',
  'Surprise': 'mean',
  'Surprise_pct': 'mean'
  }).reset_index()

  # Pivot the DataFrame for each column we want to reshape
  pivot_actual_release = aggregated_df.pivot(index='ECO_RELEASE_DT', columns='Index', values='ACTUAL_RELEASE')
  pivot_survey_median = aggregated_df.pivot(index='ECO_RELEASE_DT', columns='Index', values='BN_SURVEY_MEDIAN')
  pivot_surprise = aggregated_df.pivot(index='ECO_RELEASE_DT', columns='Index', values='Surprise')
  pivot_surprise_pct = aggregated_df.pivot(index='ECO_RELEASE_DT', columns='Index', values='Surprise_pct')

  # Flatten the MultiIndex columns and trim the 'Index' part of the name
  pivot_actual_release.columns = [f'ACTUAL_RELEASE_{col.replace(" Index", "")}' for col in pivot_actual_release.columns]
  pivot_survey_median.columns = [f'SURVEY_MEDIAN_{col.replace(" Index", "")}' for col in pivot_survey_median.columns]
  pivot_surprise.columns = [f'SURPRISE_{col.replace(" Index", "")}' for col in pivot_surprise.columns]
  pivot_surprise_pct.columns = [f'SURPRISE_PCT_{col.replace(" Index", "")}' for col in pivot_surprise_pct.columns]

  # Now, let's merge the pivoted data with the original DataFrame
  merged_df = df.join(pivot_actual_release, on='ECO_RELEASE_DT')
  merged_df = merged_df.join(pivot_survey_median, on='ECO_RELEASE_DT')
  merged_df = merged_df.join(pivot_surprise, on='ECO_RELEASE_DT')
  merged_df = merged_df.join(pivot_surprise_pct, on='ECO_RELEASE_DT')

  return merged_df
  
def get_only_macro_interpolated_v1(df, IMPUTE_STRATEGY, col_list):

  # Ensure the date column is in datetime format
  df['ECO_RELEASE_DT'] = pd.to_datetime(df['ECO_RELEASE_DT'])

  # Set the date column as the index and remove any duplicate dates
  df.set_index('ECO_RELEASE_DT', inplace=True)
  df = df[~df.index.duplicated(keep='first')]

  # Create a date range for reindexing from the earliest to the latest date
  date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')

  # Reindex the DataFrame to include all dates in the range, filling missing ones
  df = df.reindex(date_range)

  if IMPUTE_STRATEGY=='LI':
    # Interpolation methods such as linear or quadratic if appropriate
    for column in col_list:
      df.replace([np.inf, -np.inf], np.nan, inplace=True)
      df[column].interpolate(method='linear', inplace=True)
      df[column] = df[column].fillna(method='ffill').fillna(method='bfill')

  if IMPUTE_STRATEGY=='RWM':
    window_size = 7  # Window size for rolling mean

    # Interpolation methods such as linear or quadratic if appropriate
    ## EUR/USD
    for column in col_list:
      df.replace([np.inf, -np.inf], np.nan, inplace=True)
      df[column].fillna(df[column].rolling(window=window_size, min_periods=1).mean(), inplace=True)
      df[column] = df[column].fillna(method='ffill').fillna(method='bfill')

  return df
  
def get_interpolated_macro_df_v3(df, date_col, indices, IMPUTE_STRATEGY):
  # Create an empty DataFrame to hold the results
  result_df = pd.DataFrame()

  for index_name in indices:
    # Filter the DataFrame for the current index
    temp_df = df[df['Index'] == index_name].copy()

    # Ensure the date column is in datetime format
    temp_df[date_col] = pd.to_datetime(temp_df[date_col])

    # Set the date column as the index and remove any duplicate dates
    temp_df.set_index(date_col, inplace=True)
    temp_df = temp_df[~temp_df.index.duplicated(keep='first')]

    # Create a date range for reindexing from the earliest to the latest date
    date_range = pd.date_range(start=temp_df.index.min(), end=temp_df.index.max(), freq='D')

    # Reindex the DataFrame to include all dates in the range, filling missing ones
    temp_df = temp_df.reindex(date_range)

    # Interpolate missing values in the index column
    if IMPUTE_STRATEGY == 'LI':
      temp_df[index_name] = temp_df[index_name].interpolate(method='linear')
    elif IMPUTE_STRATEGY == 'RWM':
      window_size = 7  # Define the window size for rolling mean
      temp_df[index_name].fillna(temp_df[index_name].rolling(window=window_size, min_periods=1).mean(), inplace=True)

    # Reset the index to convert the 'Date' index back into a column
    temp_df.reset_index(inplace=True)

    # Rename the index column back to 'Date'
    temp_df.rename(columns={'index': 'Date'}, inplace=True)

  # Add the interpolated data to the result DataFrame
  result_df = pd.concat([result_df, temp_df], axis=1)

  # Add the 'Economic_zone' column
  result_df['Economic_zone'] = 'Eurozone'

  return result_df
  
  
# Modified function to handle each index separately and check for column existence
def get_interpolated_macro_df_v3(df, date_col, indices, IMPUTE_STRATEGY):
  # Create an empty DataFrame to hold the results
  result_df = pd.DataFrame()

  for index_name in indices:
    # Filter the DataFrame for the current index
    temp_df = df[df['Index'] == index_name].copy()

    # Ensure the date column is in datetime format
    temp_df[date_col] = pd.to_datetime(temp_df[date_col])

    # Set the date column as the index and remove any duplicate dates
    temp_df.set_index(date_col, inplace=True)
    temp_df = temp_df[~temp_df.index.duplicated(keep='first')]

    # Create a date range for reindexing from the earliest to the latest date
    date_range = pd.date_range(start=temp_df.index.min(), end=temp_df.index.max(), freq='D')

    # Reindex the DataFrame to include all dates in the range, filling missing ones
    temp_df = temp_df.reindex(date_range)

    # Check if the index_name exists as a column before interpolating
    if index_name in temp_df.columns:
      # Interpolate missing values in the index column
      if IMPUTE_STRATEGY == 'LI':
        temp_df[index_name] = temp_df[index_name].interpolate(method='linear')
      elif IMPUTE_STRATEGY == 'RWM':
        window_size = 7  # Define the window size for rolling mean
        temp_df[index_name].fillna(temp_df[index_name].rolling(window=window_size, min_periods=1).mean(), inplace=True)

    # Reset the index to convert the 'Date' index back into a column
    temp_df.reset_index(inplace=True)

    # Rename the index column back to 'Date'
    temp_df.rename(columns={'index': 'Date'}, inplace=True)

  # Add the interpolated data to the result DataFrame
  result_df = pd.concat([result_df, temp_df], axis=1)

  # Add the 'Economic_zone' column
  result_df['Economic_zone'] = 'Eurozone'

  return result_df
  
def get_interpolated_macro_df_v4(df, date_col, indices, IMPUTE_STRATEGY):
  # Create an empty DataFrame to hold the results
  result_df = pd.DataFrame()

  for index_name in indices:
    # Filter the DataFrame for the current index
    temp_df = df[df['Index'] == index_name].copy()

    # Ensure the date column is in datetime format
    temp_df[date_col] = pd.to_datetime(temp_df[date_col])

    # Set the date column as the index and remove any duplicate dates
    temp_df.set_index(date_col, inplace=True)
    temp_df = temp_df[~temp_df.index.duplicated(keep='first')]

    # Create a date range for reindexing from the earliest to the latest date
    date_range = pd.date_range(start=temp_df.index.min(), end=temp_df.index.max(), freq='D')

    # Reindex the DataFrame to include all dates in the range, filling missing ones
    temp_df = temp_df.reindex(date_range)

    # Interpolate missing values for all numeric columns
    numeric_columns = temp_df.select_dtypes(include=['number']).columns
    for column in numeric_columns:
      if IMPUTE_STRATEGY == 'LI':
        temp_df[column] = temp_df[column].interpolate(method='linear')
      elif IMPUTE_STRATEGY == 'RWM':
        window_size = 7  # Define the window size for rolling mean
        temp_df[column].fillna(temp_df[column].rolling(window=window_size, min_periods=1).mean(), inplace=True)

    # Reset the index to convert the 'Date' index back into a column
    temp_df.reset_index(inplace=True)

    # Rename the index column back to 'Date'
    temp_df.rename(columns={'index': 'Date'}, inplace=True)

  # Add the interpolated data to the result DataFrame
  result_df = pd.concat([result_df, temp_df], axis=1)

  # Add the 'Economic_zone' column
  result_df['Economic_zone'] = 'Eurozone'

  return result_df
  
def get_interpolated_macro_df_v2(df, date_col, index_name, IMPUTE_STRATEGY):
  # Ensure the date column is in datetime format
  df[date_col] = pd.to_datetime(df[date_col])

  # Set the date column as the index and remove any duplicate dates
  df.set_index(date_col, inplace=True)
  df = df[~df.index.duplicated(keep='first')]

  # Create a date range for reindexing from the earliest to the latest date
  date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')

  # Reindex the DataFrame to include all dates in the range, filling missing ones
  df = df.reindex(date_range)

  # Interpolate missing values in the index column
  # # Forward fill to propagate the last valid observation forward
  # df[index_name] = df[index_name].ffill()

  # Optionally, you can also backfill to propagate the next valid observation backward
  # df[index_name] = df[index_name].bfill()

  if IMPUTE_STRATEGY=='LI':
    # Or use interpolation methods such as linear or quadratic if appropriate
    df[index_name] = df[index_name].interpolate(method='linear')

  if IMPUTE_STRATEGY=='RWM':
    window_size = 7  # Define the window size for rolling mean
    # Apply rolling window mean to each column in the list
    df[index_name].fillna(df[index_name].rolling(window=window_size, min_periods=1).mean(), inplace=True)

  # Reset the index to convert the 'Date' index back into a column
  df.reset_index(inplace=True)
  # Rename the index column back to 'Date'
  df.rename(columns={'index': 'Date'}, inplace=True)

  return df
  
def resample_exchange_rate(fx_rate):
  # Resample FX rates to weekly/monthly averages if needed
  fx_rate.set_index('Date', inplace=True)
  daily = fx_rate.resample('D').mean()  # Daily average
  weekly = fx_rate.resample('W').mean()  # Weekly average
  monthly = fx_rate.resample('M').mean()  # Monthly average

  return daily, weekly, monthly
  
def get_only_macro_interpolated(df, base_currency, IMPUTE_STRATEGY, col_list):
  # Rename columns to reflect daily data
  surprise_name = 'Surprise_' + base_currency
  surprise_pct_name = 'Surprise_pct_' + base_currency

  index_name = 'Index_' + base_currency
  release_name = 'ACTUAL_RELEASE_' + base_currency
  survey_name = 'BN_SURVEY_MEDIAN_' + base_currency

  df.rename(columns={'Surprise': surprise_name, 'Surprise_pct': surprise_pct_name, 'Index': index_name, 'ACTUAL_RELEASE': release_name, 'BN_SURVEY_MEDIAN': survey_name}, inplace=True)

  if IMPUTE_STRATEGY=='LI':
    # Interpolation methods such as linear or quadratic if appropriate
    for column in col_list:
      df.replace([np.inf, -np.inf], np.nan, inplace=True)
      df[column].interpolate(method='linear', inplace=True)

  if IMPUTE_STRATEGY=='RWM':
    window_size = 7  # Window size for rolling mean

    # Interpolation methods such as linear or quadratic if appropriate
    ## EUR/USD
    for column in col_list:
      df.replace([np.inf, -np.inf], np.nan, inplace=True)
      df[column].fillna(df[column].rolling(window=window_size, min_periods=1).mean(), inplace=True)

  return df
  
def get_macro_and_fx_rate_daily_interpolated_v2(macro_with_sf, fx_daily_data, target_pair):
  # Convert 'ECO_RELEASE_DT' to datetime and create a 'Date' column for macro_with_sf
  macro_with_sf['Date'] = pd.to_datetime(macro_with_sf['ECO_RELEASE_DT'])
  #print("Columns after creating 'Date' in macro_with_sf:", macro_with_sf.columns)

  # Reset the index for fx_daily_data to convert the 'Date' index into a column
  fx_daily_data.reset_index(inplace=True)
  # Ensure both 'Date' columns are in datetime format
  fx_daily_data['Date'] = pd.to_datetime(fx_daily_data['Date'])
  #print("Columns after resetting index in fx_daily_data:", fx_daily_data.columns)

  # # Print out the min and max dates to check for NaT values
  # print("Macro data start date:", macro_with_sf['Date'].min())
  # print("Macro data end date:", macro_with_sf['Date'].max())
  # print("FX data start date:", fx_daily_data['Date'].min())
  # print("FX data end date:", fx_daily_data['Date'].max())

  # Check if there are still NaT values after conversion
  if macro_with_sf['Date'].isnull().any() or fx_daily_data['Date'].isnull().any():
    raise ValueError("The data contains NaT in date columns, please check your data.")

  # Now find the common start and end dates
  common_start = max(macro_with_sf['Date'].min(), fx_daily_data['Date'].min())
  common_end = min(macro_with_sf['Date'].max(), fx_daily_data['Date'].max())

  # Filter the dataframes to the common date range
  macro_with_sf = macro_with_sf[(macro_with_sf['Date'] >= common_start) & (macro_with_sf['Date'] <= common_end)]
  fx_daily_data = fx_daily_data[(fx_daily_data['Date'] >= common_start) & (fx_daily_data['Date'] <= common_end)]

  # Interpolate macroeconomic data
  macro_df_interpolated = get_interpolated_macro_df_v2(macro_with_sf, 'Date', 'Index', "LI")

  # Merge the datasets using merge_asof with a tolerance for matching dates
  merged_df = pd.merge_asof(fx_daily_data.sort_values('Date'), macro_df_interpolated.sort_values('Date'), on='Date', tolerance=pd.Timedelta('1 day'))

  # Feature engineering: create moving averages or changes over time
  merged_df[target_pair + '_1d_MA'] = merged_df[target_pair].rolling(window=1).mean()  # 1-day moving average
  merged_df[target_pair + '_1d_Change'] = merged_df[target_pair].diff(1)  # Change over 1 day

  merged_df[target_pair + '_3d_MA'] = merged_df[target_pair].rolling(window=3).mean()  # 3-day moving average
  merged_df[target_pair + '_3d_Change'] = merged_df[target_pair].diff(3)  # Change over 3 day

  merged_df[target_pair + '_5d_MA'] = merged_df[target_pair].rolling(window=5).mean()  # 5-day moving average
  merged_df[target_pair + '_5d_Change'] = merged_df[target_pair].diff(5)  # Change over 5 day

  #merged_df[target_pair + '_7d_MA'] = merged_df[target_pair].rolling(window=7).mean()  # 7-day moving average
  merged_df[target_pair + '_7d_Change'] = merged_df[target_pair].diff(7)  # Change over 7 days

  # Add rolling standard deviation calculations
  merged_df[target_pair + '_1d_StdDev'] = merged_df[target_pair].rolling(window=1).std()  # 1-day rolling standard deviation
  merged_df[target_pair + '_3d_StdDev'] = merged_df[target_pair].rolling(window=3).std()  # 3-day rolling standard deviation
  merged_df[target_pair + '_5d_StdDev'] = merged_df[target_pair].rolling(window=5).std()  # 5-day rolling standard deviation
  merged_df[target_pair + '_7d_StdDev'] = merged_df[target_pair].rolling(window=7).std()  # 7-day rolling standard deviation

  # Rename columns to reflect daily data
  surprise_name = 'Surprise_' + target_pair.split('/')[0]
  surprise_pct_name = 'Surprise_pct_' + target_pair.split('/')[0]

  index_name = 'Index_' + target_pair.split('/')[0]
  release_name = 'ACTUAL_RELEASE_' + target_pair.split('/')[0]
  survey_name = 'BN_SURVEY_MEDIAN_' + target_pair.split('/')[0]

  merged_df.rename(columns={'Surprise': surprise_name, 'Surprise_pct': surprise_pct_name, 'Index': index_name, 'ACTUAL_RELEASE': release_name, 'BN_SURVEY_MEDIAN': survey_name}, inplace=True)

  return merged_df
  
def interpolate_combined_data(df, main_columns_to_interpolate, additional_columns_to_interpolate, IMPUTE_STRATEGY):
  if IMPUTE_STRATEGY=='LI':
    # Interpolation methods such as linear or quadratic if appropriate
    ## EUR/USD
    for column in main_columns_to_interpolate:
      df[column].interpolate(method='linear', inplace=True)
    for column in additional_columns_to_interpolate:
      df.replace([np.inf, -np.inf], np.nan, inplace=True)
      df[column].interpolate(method='linear', inplace=True)

  if IMPUTE_STRATEGY=='RWM':
    window_size = 7  # Window size for rolling mean

    # Interpolation methods such as linear or quadratic if appropriate
    ## EUR/USD
    for column in main_columns_to_interpolate:
      df[column].fillna(df[column].rolling(window=window_size, min_periods=1).mean(), inplace=True)
    for column in additional_columns_to_interpolate:
      df.replace([np.inf, -np.inf], np.nan, inplace=True)
      df[column].fillna(df[column].rolling(window=window_size, min_periods=1).mean(), inplace=True)

  return df