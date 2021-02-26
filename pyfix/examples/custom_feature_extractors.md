```python
from pyfix import extract, feature, legacy, io, clean
from pyfix.extract import ICA, NiftiMap, TimeCourse, NiftiTimeCourse
from pyfix.feature import feature_extractor
import os.path as op
import numpy as np
from fsl.data import featanalysis
import nibabel as nb
```

# Download example data


```python
datadir = '/Users/seanf/scratch/testing-pyfix/example_data'
```

# Extract features (single subject)

### Import data

If your fMRI data is in a confirming `feat` directory, you can simply import it into a `FixData` object:


```python
%%time

featdir = f'{datadir}/Rest_MB6.feat'

d = extract.FixData.from_melodic_dir(
    featdir,
    fixdir=op.join(featdir, 'fix')
)
```

    CPU times: user 98.2 ms, sys: 14.3 ms, total: 113 ms
    Wall time: 123 ms


### Create custom feature extractors

Feature extractor functions:
- must be decorated with `@feature_extractor`.
- must have the function signature `def function_name(data, ic_idx):`
- support the following output types:
    1. scalar
    2. `list`
    3. `dict`
    4. `pandas.Series`

For example:


```python
@feature_extractor
def mean_motion_parameters(data, ic_idx):
    
    # get motion parameters from FixData object
    mp = data.timecourses['motparams'].data
    
    # calculate mean tr and rot
    mp = {
        'mean_tr': np.mean(mp[:, :3]),
        'mean_rot': np.mean(mp[:, 3:]),
    }

    # return as dict
    return mp

@feature_extractor
def ic_max(data, ic_idx):
    
    # get ic_timeseries
    ic_timeseries = data.ica.get_timecourse(ic_idx)
    
    # get ic_timeseries max
    mx = np.max(ic_timeseries)

    # return as scalar
    return mx
```

### Extract Features

Extract custom features from `FixData` object:


```python
%%time

f = extract.extract_features(
    data=d,
    features=['mean_motion_parameters', 'ic_max'],
)

f.head()
```

    CPU times: user 102 ms, sys: 1.83 ms, total: 104 ms
    Wall time: 103 ms





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_tr</th>
      <th>mean_rot</th>
      <th>ic_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.394598e-17</td>
      <td>-4.282958e-18</td>
      <td>2.157280</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.394598e-17</td>
      <td>-4.282958e-18</td>
      <td>2.655940</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.394598e-17</td>
      <td>-4.282958e-18</td>
      <td>1.518387</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.394598e-17</td>
      <td>-4.282958e-18</td>
      <td>2.499870</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.394598e-17</td>
      <td>-4.282958e-18</td>
      <td>2.653833</td>
    </tr>
  </tbody>
</table>
</div>



### Existing feature extractors

There are numerous legacy features extractors within `pyFIX`.  They can be listed with the following command:


```python
legacy.LEGACY_FEATURE_EXTRACTORS
```




    ['arfull',
     'arvswgn',
     'clusterdist',
     'dim',
     'edgemasks',
     'entropy',
     'everynthvariance',
     'fftcoarse',
     'fftfiner',
     'fftfinerwrtnull',
     'kurtosis',
     'masktscorrandoverlap',
     'maxtfce',
     'mean_median',
     'motioncorrelation',
     'negativevspositive',
     'nic',
     'oupjk',
     'pixdim',
     'sagmasks',
     'skewness',
     'slicewisestats',
     'smoothest',
     'spatialoverlap',
     'stripe',
     'tsjump',
     'zstattofuncratio']




```python

```
