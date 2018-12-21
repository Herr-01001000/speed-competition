import numpy as np
import pandas as pd


def pandas_update(state, root_cov, measurement, loadings, meas_var):
    """Update *state* and *root_cov* with with a *measurement*.

    Args:
        state (pd.Series): pre-update estimate of the unobserved state vector
        root_cov (pd.DataFrame): lower triangular matrix square-root of the
            covariance matrix of the state vector before the update
        measurement (float): the measurement to incorporate
        loadings (pd.Series): the factor loadings
        meas_var(float): variance of the measurement error
    Returns:
        updated_state (pd.Series)
        updated_root_cov (pd.DataFrame)

    """
    expected_measurement = state.dot(loadings)
    residual = measurement - expected_measurement
    f_star = root_cov.T.dot(loadings)
    first_row = pd.DataFrame(
        data=[np.sqrt(meas_var)] + [0] * len(loadings),
        index=[0] + list(state.index)).T
    other_rows = pd.concat([f_star, root_cov.T], axis=1)
    m = pd.concat([first_row, other_rows])
    r = np.linalg.qr(m, mode='r')
    root_sigma = r[0, 0]
    kalman_gain = pd.Series(r[0, 1:], index=state.index) / root_sigma
    updated_root_cov = pd.DataFrame(
        data=r[1:, 1:],
        columns=state.index,
        index=state.index,
    ).T
    updated_state = state + kalman_gain * residual

    return updated_state, updated_root_cov


def pandas_batch_update(states, root_covs, measurements, loadings, meas_var):
    """Call pandas_update repeatedly.

    Args:
        states (pd.DataFrame)
        root_covs (list)
        measurements (pd.Series)
        loadings (pd.Series)
        meas_var (float)
    Returns:
        updated_states (pd.DataFrame)
        updated_root_covs (list)

    """
    out_states = []
    out_root_covs = []
    for i in range(len(states)):
        updated_state, updated_root_cov = pandas_update(
            state=states.loc[i],
            root_cov=root_covs[i],
            measurement=measurements[i],
            loadings=loadings,
            meas_var=meas_var
        )
        out_states.append(updated_state)
        out_root_covs.append(updated_root_cov)
    out_states = pd.concat(out_states, axis=1).T
    return out_states, out_root_covs
