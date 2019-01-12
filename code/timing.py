import numpy as np
import pandas as pd
from time import time
from update import pandas_batch_update
from update import fast_batch_update_np
from update import fast_batch_update_nb
from update import fast_batch_update_tf
from update_cl import fast_batch_update_cl

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

# time the function
#runtimes = []
#for i in range(2):
#    start = time()
#    out_states, out_root_covs = pandas_batch_update(
#        states=states_pd,
#        root_covs=root_covs_pd,
#        measurements=meas_bwght_pd,
#        loadings=loadings_bwght_pd,
#        meas_var=meas_var_bwght,
#    )
#    stop = time()
#    runtimes.append(stop - start)
##    print("pandas_batch_update took {} seconds.".format(runtimes))
#
#print("pandas_batch_update took {} seconds.".format(np.mean(runtimes)))


# time the function
runtimes = []
for i in range(20):
    start = time()
    out_states_fast, out_root_covs_fast = fast_batch_update_np(
        states=states_np,
        root_covs=root_covs_np,
        measurements=meas_bwght_np,
        loadings=loadings_bwght_np,
        meas_var=meas_var_bwght,
    )
    stop = time()
    runtimes.append(stop - start)
    print("fast_batch_update_np took {} seconds.".format(runtimes))

print("fast_batch_update_np took {} seconds.".format(np.mean(runtimes)))


# time the function
runtimes = []
for i in range(20):
    start = time()
    out_states_fast, out_root_covs_fast = fast_batch_update_nb(
        states=states_np,
        root_covs=root_covs_np,
        measurements=meas_bwght_np,
        loadings=loadings_bwght_np,
        meas_var=meas_var_bwght,
    )
    stop = time()
    runtimes.append(stop - start)
    print("fast_batch_update_nb took {} seconds.".format(runtimes))

print("fast_batch_update_nb took {} seconds.".format(np.mean(runtimes)))


# time the function
runtimes = []
for i in range(20):
    start = time()
    out_states_fast, out_root_covs_fast = fast_batch_update_tf(
        states=states_np,
        root_covs=root_covs_np,
        measurements=meas_bwght_np,
        loadings=loadings_bwght_np,
        meas_var=meas_var_bwght,
    )
    stop = time()
    runtimes.append(stop - start)
    print("fast_batch_update_tf took {} seconds.".format(runtimes))

print("fast_batch_update_tf took {} seconds.".format(np.mean(runtimes)))
