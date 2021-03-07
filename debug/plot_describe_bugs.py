import pandas as pd

from collections import Counter 
import numpy as np

df = pd.read_csv('data/log_compute_rest_spectra.csv')
df['perc_epochs'] = df['n_epochs'] / df['n_events']
df['perc_epochs'][np.isnan(df['perc_epochs'])] = 0

from collections import Counter
error_counter = Counter(df.error)
# gives us:
"""
Counter({'None': 607,
         'TypeError("\'NoneType\' object is not iterable",)': 12,
         "IndexError('boolean index did not match indexed array along dimension 0; dimension is 0 but corresponding boolean dimension is 4097',)": 22,
         "RuntimeError('EEG 61 or EEG 62 channel not found !!',)": 2})
"""


"""
1. None error
-------------

Projection is None because it does not have enough events.
Fix -> avoid setting projection

2. bool error
-------------

fix autoreject


3. EOG channels not declared
----------------------------

skip EOG part of pipeline

"""