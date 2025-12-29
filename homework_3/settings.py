AllColumns = ['Date', 'BC','N_CPC','PM-10', 'PM-2.5', 'PM-1.0', 'NO2', 'O3', 'SO2', 'CO', 'NO', 'NOX', 'TEMP', 'HUM']
FeatureColumns = ['N_CPC','PM-10', 'PM-2.5', 'PM-1.0', 'NO2', 'O3', 'SO2', 'CO', 'NO', 'NOX', 'TEMP', 'HUM']
DateColumn = 'Date'
RefColumn = 'BC'
CrossValidationTrainPercentage = 0.7
DatasetFile = 'dataset/BC-Data-Set.csv'


# Feature Selection
ReducedFeatureColumns = ['N_CPC','PM-10', 'PM-2.5', 'PM-1.0', 'NO2', 'O3', 'NO', 'TEMP', 'HUM']