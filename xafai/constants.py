"""
This file provides the exact names of the feature columns, target column and fairness configuration.

FEATURES, TARGET and CONFIG_FAI are required in the report.
"""
FEATURES = [f"feat_{i}" for i in range(1, 94)]

TARGET = 'target'

# List privileged info
CONFIG_FAI = {
    'feat_1': {
        'unprivileged_attribute_values': [0],
        'privileged_attribute_values': [1],
    },
}
