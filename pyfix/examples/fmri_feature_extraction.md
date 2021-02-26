```python
from pyfix import extract, feature, legacy, io, clean
from pyfix.extract import ICA, NiftiMap, TimeCourse, NiftiTimeCourse
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

    CPU times: user 95.9 ms, sys: 12.4 ms, total: 108 ms
    Wall time: 120 ms


Alternatively, you can manually construct a `FixData` object:


```python
%%time

meldir = f'{datadir}/Rest_MB6.feat'

fixdir = op.join(meldir, 'pyfix')

# load melodic data

func = nb.load(op.join(meldir, 'filtered_func_data.nii.gz'))
tr = func.header.get_zooms()[3]
func_brainmask = nb.load(op.join(meldir, 'mask.nii.gz'))

ic_timeseries = np.loadtxt(op.join(meldir, 'filtered_func_data.ica', 'melodic_mix'))
ic_map = nb.load(op.join(meldir, 'filtered_func_data.ica', 'melodic_IC.nii.gz'))

struct_fast_seg = nb.load(op.join(meldir, 'reg/highres_pveseg.nii.gz'))
struct = nb.load(op.join(meldir, 'reg/highres.nii.gz'))
struct2func_xfm = op.join(meldir, 'reg/highres2example_func.mat')

standard = op.join(meldir, 'reg/standard.nii.gz')
std2func_xfm = op.join(meldir, 'reg/standard2example_func.mat')

# highpass settings
fsf = featanalysis.loadSettings(meldir)
highpass_fwhm = float(fsf['paradigm_hp'])

# filter, and normalise motion confounds, as required

mp = op.join(meldir, 'mc', 'prefiltered_func_data_mcf_conf_hp.nii.gz')
mp = clean.motion_confounds(mp, tr=tr, temporal_fwhm=highpass_fwhm, tmpdir=meldir)
mp = clean._normalise(mp)

# instantiate FixData class

d = extract.FixData(
    fixdir=fixdir,
    ica=ICA(
        maps=NiftiMap(ic_map),
        timecourses=TimeCourse(ic_timeseries, fs=1/tr),
    ),
    spatialmaps={
        'dseg': NiftiMap(struct_fast_seg, xfm=struct2func_xfm),
        'brainmask': NiftiMap(func_brainmask),
        'struct': NiftiMap(struct, xfm=struct2func_xfm),
        'standard': NiftiMap(standard, xfm=std2func_xfm)
    },
    timecourses={
        'motparams': TimeCourse(mp, fs=1/tr),
    },
    hp_fwhm=highpass_fwhm,
    input=NiftiTimeCourse(func, fs=1/tr),
)
```

    CPU times: user 86.8 ms, sys: 4.48 ms, total: 91.3 ms
    Wall time: 90.4 ms


### Extract Features

Extract all `legacy FIX` features from `FixData` object:


```python
%%time

f = extract.extract_features(
    data=d,
    features=legacy.LEGACY_FEATURE_EXTRACTORS,
)

# save to disk (optional)
io.save_features(op.join(fixdir, 'features'), f)
```

    CPU times: user 8min 34s, sys: 58.4 s, total: 9min 32s
    Wall time: 7min 4s



```python
! ls {fixdir}/features.pyfix_features
```

    /Users/seanf/scratch/testing-pyfix/example_data/Rest_MB6.feat/pyfix/features.pyfix_features



```python
f.head()
```




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
      <th>arfull:0</th>
      <th>arfull:1</th>
      <th>arfull:2</th>
      <th>arfull:3</th>
      <th>arfull:4</th>
      <th>arvswgn:0</th>
      <th>arvswgn:1</th>
      <th>clusterdist:0</th>
      <th>clusterdist:1</th>
      <th>clusterdist:2</th>
      <th>...</th>
      <th>tsjump:0</th>
      <th>tsjump:1</th>
      <th>tsjump:2</th>
      <th>tsjump:3</th>
      <th>tsjump:4</th>
      <th>tsjump:5</th>
      <th>zstattofuncratio:0</th>
      <th>zstattofuncratio:1</th>
      <th>zstattofuncratio:2</th>
      <th>zstattofuncratio:3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.077165</td>
      <td>-0.960592</td>
      <td>0.076100</td>
      <td>-0.847741</td>
      <td>-0.117481</td>
      <td>-0.003071</td>
      <td>0.076653</td>
      <td>92.0</td>
      <td>239.000000</td>
      <td>21428.0</td>
      <td>...</td>
      <td>1.320846</td>
      <td>4.748875</td>
      <td>0.077263</td>
      <td>6.193506</td>
      <td>0.007890</td>
      <td>0.961338</td>
      <td>15608.197334</td>
      <td>7942.848584</td>
      <td>0.000750</td>
      <td>0.000265</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.261969</td>
      <td>-0.858894</td>
      <td>0.240919</td>
      <td>-0.615424</td>
      <td>-0.283469</td>
      <td>-0.021162</td>
      <td>0.247240</td>
      <td>111.0</td>
      <td>138.540541</td>
      <td>14854.0</td>
      <td>...</td>
      <td>1.781619</td>
      <td>3.377118</td>
      <td>0.277973</td>
      <td>4.277896</td>
      <td>0.005450</td>
      <td>0.860723</td>
      <td>11339.684814</td>
      <td>6843.723486</td>
      <td>0.000593</td>
      <td>0.000187</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.211158</td>
      <td>-0.888017</td>
      <td>0.205263</td>
      <td>-0.739640</td>
      <td>-0.167087</td>
      <td>-0.005726</td>
      <td>0.209917</td>
      <td>142.0</td>
      <td>70.943662</td>
      <td>8629.0</td>
      <td>...</td>
      <td>2.797601</td>
      <td>6.204240</td>
      <td>0.203113</td>
      <td>9.606800</td>
      <td>0.012222</td>
      <td>0.897422</td>
      <td>13009.734316</td>
      <td>5120.866943</td>
      <td>0.000444</td>
      <td>0.000110</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.260129</td>
      <td>-0.859966</td>
      <td>0.244080</td>
      <td>-0.646357</td>
      <td>-0.248392</td>
      <td>-0.013731</td>
      <td>0.248810</td>
      <td>176.0</td>
      <td>36.897727</td>
      <td>4530.0</td>
      <td>...</td>
      <td>1.724502</td>
      <td>3.341166</td>
      <td>0.266092</td>
      <td>4.201665</td>
      <td>0.005346</td>
      <td>0.866041</td>
      <td>11032.543867</td>
      <td>4485.424512</td>
      <td>0.000474</td>
      <td>0.000147</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.838972</td>
      <td>-0.399956</td>
      <td>0.795287</td>
      <td>-0.308691</td>
      <td>-0.228186</td>
      <td>-0.097000</td>
      <td>0.797256</td>
      <td>178.0</td>
      <td>62.235955</td>
      <td>8884.0</td>
      <td>...</td>
      <td>2.932550</td>
      <td>2.689752</td>
      <td>1.187185</td>
      <td>3.264599</td>
      <td>0.004159</td>
      <td>0.402556</td>
      <td>16305.365244</td>
      <td>5060.037573</td>
      <td>0.000943</td>
      <td>0.000193</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 186 columns</p>
</div>


