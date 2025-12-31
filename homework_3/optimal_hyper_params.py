non_shuffle_hyper_params = {
    2: {
        'n_estimators': 50,
        'max_depth': 3,
        'min_samples_split': 2,
        'learning_rate': 0.1,
    },
    4: {
        'n_estimators': 450,
        'max_depth': 4,
        'min_samples_split': 2,
        'learning_rate': 0.15,
    },
    6: {
        'n_estimators': 500,
        'max_depth': 4,
        'min_samples_split': 4,
        'learning_rate': 0.05,
    },
    8: {
        'n_estimators': 450,
        'max_depth': 2,
        'min_samples_split': 4,
        'learning_rate': 0.05,
    },
    16: {
        'n_estimators': 500,
        'max_depth': 3,
        'min_samples_split': 2,
        'learning_rate': 0.05,
    },
    0.3: {
        'n_estimators': 250,
        'max_depth': 4,
        'min_samples_split': 3,
        'learning_rate': 0.05,
    },
    0.5: {
        'n_estimators': 250,
        'max_depth': 2,
        'min_samples_split': 4,
        'learning_rate': 0.15,
    },
    0.7: {
        'n_estimators': 300,
        'max_depth': 3,
        'min_samples_split': 3,
        'learning_rate': 0.05,
    },
    0.9: {
        'n_estimators': 350,
        'max_depth': 4,
        'min_samples_split': 2,
        'learning_rate': 0.05,
    }
}

non_shuffle_features = {
    0.3: ['NO2', 'N_CPC', 'HUM', 'NO', 'PM-1.0', 'SO2'],
    0.5: ['N_CPC', 'PM-1.0', 'SO2', 'NOX', 'NO2'],
    0.7: ['PM-1.0', 'CO', 'SO2', 'PM-2.5', 'TEMP', 'NO', 'HUM', 'NO2', 'N_CPC'],
    0.9: ['TEMP', 'PM-1.0', 'NOX', 'HUM', 'PM-10', 'N_CPC', 'NO2', 'SO2', 'PM-2.5']
}

non_shuffle_features_lasso = {
    0.3: ['N_CPC', 'PM-2.5', 'PM-1.0', 'NO2', 'O3', 'SO2', 'NOX'],
    0.5: ['N_CPC', 'PM-10', 'PM-1.0', 'NO2', 'O3', 'SO2', 'NOX'],
    0.7: ['N_CPC', 'PM-10', 'PM-1.0', 'NO2', 'SO2', 'NOX'],
    0.9: ['N_CPC', 'PM-10', 'PM-1.0', 'NO2', 'O3', 'SO2', 'CO', 'NOX']
}