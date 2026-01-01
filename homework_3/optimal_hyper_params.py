non_shuffle_hyper_params = {
    0.1: {
        'n_estimators': 150,
        'max_depth': 2,
        'min_samples_split': 4,
        'learning_rate': 0.1,
    },
    0.2: {
        'n_estimators': 250,
        'max_depth': 2,
        'min_samples_split': 7,
        'learning_rate': 0.05,
    },
    0.3: {
        'n_estimators': 150,
        'max_depth': 2,
        'min_samples_split': 2,
        'learning_rate': 0.05,
    },
    0.4: {
        'n_estimators': 100,
        'max_depth': 3,
        'min_samples_split': 5,
        'learning_rate': 0.15,
    },
    0.5: {
        'n_estimators': 200,
        'max_depth': 2,
        'min_samples_split': 3,
        'learning_rate': 0.1,
    },
    0.6: {
        'n_estimators': 100,
        'max_depth': 3,
        'min_samples_split': 5,
        'learning_rate': 0.15,
    },
    0.7: {
        'n_estimators': 200,
        'max_depth': 3,
        'min_samples_split': 2,
        'learning_rate': 0.1,
    },
    0.8: {
        'n_estimators': 200,
        'max_depth': 3,
        'min_samples_split': 4,
        'learning_rate': 0.15,
    },
    0.9: {
        'n_estimators': 450,
        'max_depth': 5,
        'min_samples_split': 3,
        'learning_rate': 0.05,
    }
}

non_shuffle_features = {
    0.1: ['NOX', 'N_CPC', 'PM-2.5'],
    0.2: ['HUM', 'NO', 'NO2', 'N_CPC', 'PM-1.0', 'SO2', 'TEMP'],
    0.3: ['HUM', 'NO2', 'NOX', 'N_CPC', 'PM-1.0', 'SO2', 'TEMP'],
    0.4: ['NO', 'NO2', 'N_CPC', 'PM-1.0', 'SO2'],
    0.5: ['HUM', 'NO2', 'NOX', 'N_CPC', 'PM-1.0', 'SO2'],
    0.6: ['HUM', 'NO', 'NO2', 'N_CPC', 'PM-1.0', 'SO2'],
    0.7: ['HUM', 'NO2', 'NOX', 'N_CPC', 'PM-1.0', 'SO2'],
    0.8: ['HUM', 'NO2', 'NOX', 'N_CPC', 'PM-2.5', 'SO2'],
    0.9: ['CO', 'HUM', 'NO2', 'NOX', 'N_CPC', 'PM-1.0', 'PM-2.5', 'SO2']
}

non_shuffle_features_lasso = {
    0.1: ['HUM', 'NO2', 'NOX', 'N_CPC', 'O3', 'PM-10', 'PM-2.5', 'SO2'],
    0.2: ['HUM', 'NO2', 'NOX', 'N_CPC', 'O3', 'PM-1.0'],
    0.3: ['NO2', 'N_CPC', 'O3', 'PM-1.0', 'PM-2.5'],
    0.4: ['NO2', 'NOX', 'N_CPC', 'O3', 'PM-1.0'],
    0.5: ['CO', 'HUM', 'NO', 'NO2', 'N_CPC', 'O3', 'PM-1.0', 'PM-10', 'PM-2.5', 'SO2', 'TEMP'],
    0.6: ['HUM', 'NO2', 'NOX', 'N_CPC', 'O3', 'PM-1.0', 'SO2'],
    0.7: ['NO2', 'NOX', 'N_CPC', 'PM-1.0', 'PM-2.5'],
    0.8: ['NO2', 'NOX', 'N_CPC', 'PM-1.0', 'PM-2.5'],
    0.9: ['HUM', 'NO2', 'NOX', 'N_CPC', 'PM-1.0', 'PM-2.5', 'SO2', 'TEMP']
}