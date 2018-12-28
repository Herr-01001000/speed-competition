import numpy as np
import pandas as pd
#from numba import jit
import tensorflow as tf


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


#@jit(nopython=True)
def fast_batch_update(states, root_covs, measurements, loadings, meas_var):
     """Update state estimates for a whole dataset.

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
     
     f_star = np.dot(root_covs.transpose((0,2,1)), loadings).reshape(len(states), len(loadings), 1)
     m_left = np.concatenate([np.full((len(states),1,1), np.sqrt(meas_var)), f_star], axis=1)
     m_right = np.concatenate([np.zeros((len(states),1,len(loadings))), root_covs.transpose((0,2,1))], axis=1)
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
     
 
def faster_batch_update(states, root_covs, measurements, loadings, meas_var):
     """Update state estimates for a whole dataset.

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
     
     root_covs = tf.constant(root_covs)
     p = tf.matmul(root_covs, root_covs, transpose_b=True)
     sess = tf.Session()
     p = sess.run(p)
     
     f = np.dot(p, loadings)
     sigmas = np.dot(f, loadings) + np.full(len(states), meas_var)
     k = f / sigmas.reshape(2207, 1)
     updated_states = states + k * residuals.reshape(2207, 1)
     
     f_tf = tf.constant(f.reshape(2207,5,1))
     f_2 = tf.matmul(f_tf, f_tf, transpose_b=True)
     f_2 = sess.run(f_2)
     
     updated_p = p - f_2 / sigmas.reshape(2207,1,1)
     
     updated_p = tf.constant(updated_p)
     l= tf.linalg.cholesky(updated_p)
     updated_root_covs = sess.run(l)
     
     return updated_states, updated_root_covs
