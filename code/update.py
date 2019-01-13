import numpy as np
import pandas as pd
import tensorflow as tf
from numba import jit


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


def fast_batch_update_np(states, root_covs, measurements, loadings, meas_var):
    """Update state estimates for a whole dataset by numpy.
    Let nstates be the number of states and nobs the number of observations.
    Args:
        states (np.ndarray): 2d array of size (nobs, nstates)
        root_covs (np.ndarray): 3d array of size (nobs, nstates, nstates)
        measurements (np.ndarray): 1d array of size (nobs)
        loadings (np.ndarray): 1d array of size (nstates)
        meas_var (float):
    Returns:
        updated_states (np.ndarray): 2d array of size (nobs, nstates)
        updated_root_covs (np.ndarray): 3d array of size (nobs, nstates, nstates)
    """
    residuals = measurements - np.dot(states, loadings)

    f_star = np.dot(root_covs.transpose((0, 2, 1)), loadings).reshape(len(states), len(loadings), 1)
    m_left = np.concatenate([np.full((len(states), 1, 1), np.sqrt(meas_var)), f_star], axis=1)
    m_right = np.concatenate([np.zeros((len(states), 1, len(loadings))), root_covs.transpose((0, 2, 1))], axis=1)
    m = np.concatenate([m_left, m_right], axis=2)

    updated_states = np.zeros((len(states), len(loadings)))
    updated_root_covs = np.zeros((len(states), len(loadings), len(loadings)))
    for i in range(len(states)):
        r = np.linalg.qr(m[i], mode='r')
        root_sigma = r[0, 0]
        kalman_gain = r[0, 1:] / root_sigma
        updated_root_covs[i] = r[1:, 1:].T
        updated_states[i] = states[i] + np.dot(kalman_gain, residuals[i])

    return updated_states, updated_root_covs


@jit(nopython=True)
def fast_batch_update(states, root_covs, measurements, loadings, meas_var):
    """Update state estimates for a whole dataset by numba.
    Let nstates be the number of states and nobs the number of observations.
    Args:
        states (np.ndarray): 2d array of size (nobs, nstates)
        root_covs (np.ndarray): 3d array of size (nobs, nstates, nstates)
        measurements (np.ndarray): 1d array of size (nobs)
        loadings (np.ndarray): 1d array of size (nstates)
        meas_var (float):
    Returns:
        updated_states (np.ndarray): 2d array of size (nobs, nstates)
        updated_root_covs (np.ndarray): 3d array of size (nobs, nstates, nstates)
    """
    residuals = measurements - np.dot(states, loadings)

    f_star = np.zeros((len(states), len(loadings), 1))
    for i in range(len(root_covs)):
        f_star[i] = np.dot(root_covs[i].T, loadings.reshape(len(loadings), 1))

    m_left = np.zeros((len(states), len(loadings)+1, 1))
    m_left[:, :1, :] = np.full((len(states), 1, 1), np.sqrt(meas_var))
    m_left[:, 1:, :] = f_star
    m_right = np.zeros((len(states), len(loadings)+1, len(loadings)))
    m_right[:, :1, :] = np.zeros((len(states), 1, len(loadings)))
    m_right[:, 1:, :] = root_covs.transpose((0, 2, 1))
    m = np.zeros((len(states), len(loadings)+1, len(loadings)+1))
    m[:, :, :1] = m_left
    m[:, :, 1:] = m_right

    updated_states = np.zeros((len(states), len(loadings)))
    updated_root_covs = np.zeros((len(states), len(loadings), len(loadings)))
    for i in range(len(states)):
        q, r = np.linalg.qr(m[i])
        root_sigma = r[0, 0]
        kalman_gain = r[0, 1:] / root_sigma
        updated_root_covs[i] = r[1:, 1:].T
        updated_states[i] = states[i] + kalman_gain * residuals[i]

    return updated_states, updated_root_covs


def fast_batch_update_tf(states, root_covs, measurements, loadings, meas_var):
    """Update state estimates for a whole dataset by TensorFlow.
    Let nstates be the number of states and nobs the number of observations.
    Args:
        states (np.ndarray): 2d array of size (nobs, nstates)
        root_covs (np.ndarray): 3d array of size (nobs, nstates, nstates)
        measurements (np.ndarray): 1d array of size (nobs)
        loadings (np.ndarray): 1d array of size (nstates)
        meas_var (float):
    Returns:
        updated_states (np.ndarray): 2d array of size (nobs, nstates)
        updated_root_covs (np.ndarray): 3d array of size (nobs, nstates, nstates)
    Notes:
        The performance of this function is not stable. The result varies
        differently depending on tensorflow version, CUDA version, cuDNN version,
        and GPU or CPU you are using. From our experiences, the best result we
        achieve is around 1000 times faster than pandas. It is similar to the
        average result after the first run from numba, but TensorFlow can have
        the similar result in every run including the first one. And the function 
        running on GPU is much faster than it on CPU. We achieved our result on
        a NVIDIA MX150 graphics card.
    The TensorFlow GPU configuration we recommand:
        TensorFlow: version 1.12.0
        CUDA: version 10.0
    """
    residuals = measurements - np.dot(states, loadings)

    f_star = np.dot(root_covs.transpose((0, 2, 1)), loadings).reshape(len(states), len(loadings), 1)
    m_left = np.concatenate([np.full((len(states), 1, 1), np.sqrt(meas_var)), f_star], axis=1)
    m_right = np.concatenate([np.zeros((len(states), 1, len(loadings))), root_covs.transpose((0, 2, 1))], axis=1)
    m = np.concatenate([m_left, m_right], axis=2)

    m_tf = tf.constant(m)
    q, r = tf.linalg.qr(m_tf)
    sess = tf.Session()
    r = sess.run(r)

    root_sigmas = r[:, 0, 0].reshape(len(states), 1)
    kalman_gains = r[:, 0, 1:] / root_sigmas
    updated_root_covs = r[:, 1:, 1:].transpose((0, 2, 1))
    updated_states = states + kalman_gains * residuals.reshape(len(states), 1)

    return updated_states, updated_root_covs
