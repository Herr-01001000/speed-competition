import pytest
import pandas as pd
import numpy as np

from update import pandas_batch_update
from update import fast_batch_update_np
from update import fast_batch_update
from update import fast_batch_update_tf

# load and prepare data
data = pd.read_stata("../chs_data.dta")
data.replace(-100, np.nan, inplace=True)
data = data.query("age == 0")
data.reset_index(inplace=True)
data = data["weightbirth"]
data.fillna(data.mean(), inplace=True)

# fix dimensions
nobs = len(data)
state_names = ["cog", "noncog", "mother_cog", "mother_noncog", "investments"]
nstates = len(state_names)

# construct initial states
states_np = np.zeros((nobs, nstates))
states_pd = pd.DataFrame(data=states_np, columns=state_names)

# construct initial covariance matrices
root_cov = np.linalg.cholesky(
    [
        [0.1777, -0.0204, 0.0182, 0.0050, 0.0000],
        [-0.0204, 0.2002, 0.0592, 0.0261, 0.0000],
        [0.0182, 0.0592, 0.5781, 0.0862, -0.0340],
        [0.0050, 0.0261, 0.0862, 0.0667, -0.0211],
        [0.0000, 0.0000, -0.0340, -0.0211, 0.0087],
    ]
)

root_covs_np = np.zeros((nobs, nstates, nstates))
root_covs_np[:] = root_cov

root_covs_pd = []
for i in range(nobs):
    root_covs_pd.append(
        pd.DataFrame(data=root_cov, columns=state_names, index=state_names)
    )

# construct measurements
meas_bwght_np = data.values
meas_bwght_pd = data

# construct loadings
loadings_bwght_np = np.array([1.0, 0, 0, 0, 0])
loadings_bwght_pd = pd.Series(loadings_bwght_np, index=state_names)

# construct the variance
meas_var_bwght = 0.8


@pytest.fixture
def setup_update():

    out = {}
    out['states'] = states_np
    out['root_covs'] = root_covs_np
    out['measurements'] = meas_bwght_np
    out['loadings'] = loadings_bwght_np
    out['meas_var'] = meas_var_bwght

    return out


out_states, out_root_covs = pandas_batch_update(
    states_pd,
    root_covs_pd,
    meas_bwght_pd,
    loadings_bwght_pd,
    meas_var_bwght,
)


@pytest.fixture
def expected_update():

    out = {}
    out['mean'] = out_states.values
    out['cov'] = pd.concat(out_root_covs).values.reshape(nobs, nstates, nstates)

    return out


def test_fast_batch_update_np_mean(setup_update, expected_update):
    calc_mean, calc_root_cov = fast_batch_update_np(**setup_update)
    assert np.allclose(calc_mean, expected_update['mean'])


def test_fast_batch_update_np_cov_values(setup_update, expected_update):
    calc_mean, calc_root_cov = fast_batch_update_np(**setup_update)
    assert np.allclose(calc_root_cov, expected_update['cov'])


def test_fast_batch_update_mean(setup_update, expected_update):
    calc_mean, calc_root_cov = fast_batch_update(**setup_update)
    assert np.allclose(calc_mean, expected_update['mean'])


def test_fast_batch_update_cov_values(setup_update, expected_update):
    calc_mean, calc_root_cov = fast_batch_update(**setup_update)
    assert np.allclose(calc_root_cov, expected_update['cov'])


def test_fast_batch_update_tf_mean(setup_update, expected_update):
    calc_mean, calc_root_cov = fast_batch_update_tf(**setup_update)
    assert np.allclose(calc_mean, expected_update['mean'])


def test_fast_batch_update_tf_cov_values(setup_update, expected_update):
    calc_mean, calc_root_cov = fast_batch_update_tf(**setup_update)
    assert np.allclose(calc_root_cov, expected_update['cov'])
