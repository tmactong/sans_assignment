# Calibration
AllColumns = ['Date', 'WE_O3','AE_O3','WE_NO2', 'AE_NO2', 'Temp', 'RelHum', 'RefSt_O3']
FeatureColumns = ['WE_O3', 'AE_O3', 'WE_NO2', 'AE_NO2', 'Temp', 'RelHum']
DateColumn = 'Date'
RefColumn = 'RefSt_O3'
PlotHours = 4 * 7 * 24
CrossValidationTrainPercentage = 0.7
CVKNNArguments = ['k_neighbors', 'weights']
CVRFArguments = ['n_estimators', 'max_features','max_depth', 'min_samples_split']


# Denoising

# MatrixXColumns = 90
