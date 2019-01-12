import numpy as np
import pyopencl as cl
import tensorflow as tf


def faster_batch_update_tf(states, root_covs, measurements, loadings, meas_var):
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


def fast_batch_update_cl(states, root_covs, measurements, loadings, meas_var):
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
     
     ctx = cl.create_some_context()
     queue = cl.CommandQueue(ctx)
     
     mf = cl.mem_flags
     a_mat = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m)
     
     prg = cl.Program(ctx, """
     __kernel void qr(__local float *u_vec, __global float *a_mat,
         __global float *q_mat, __global float *p_mat,
         __global float *prod_mat) {
             local float u_length_squared, dot;
             float prod, vec_length = 0.0f;
             int id = get_local_id(0);
             int num_cols = get_global_size(0);
             u_vec[id] = a_mat[id*num_cols];
             barrier(CLK_LOCAL_MEM_FENCE);
             if(id == 0) {
                     for(int i=1; i<num_cols; i++) {
                             vec_length += u_vec[i] * u_vec[i];
                             }
                     u_length_squared = vec_length;
                     vec_length = sqrt(vec_length +
                                       u_vec[0] * u_vec[0]);
                                       a_mat[0] = vec_length;
                                       u_vec[0] -= vec_length;
                                       u_length_squared += u_vec[0] * u_vec[0];
                                       }
                     else {
                             a_mat[id*num_cols] = 0.0f;
                        }
            barrier(CLK_GLOBAL_MEM_FENCE);
            for(int i=1; i<num_cols; i++) {
                dot = 0.0f;
                if(id == 0) {
                    for(int j=0; j<num_cols; j++) {
                        dot += a_mat[j*num_cols + i] * u_vec[j];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                a_mat[id*num_cols + i] -= 2 * u_vec[id] *
                dot / u_length_squared;
            }
            for(int i=0; i<num_cols; i++) {
                q_mat[id*num_cols + i] = -2 * u_vec[i] *
                u_vec[id] / u_length_squared;
            }
            q_mat[id*num_cols + id] += 1;
            barrier(CLK_GLOBAL_MEM_FENCE);
            for(int col = 1; col < num_cols-1; col++) {
                u_vec[id] = a_mat[id * num_cols + col];
                barrier(CLK_LOCAL_MEM_FENCE);
                if(id == col) {
                vec_length = 0.0f;
                for(int i = col + 1; i < num_cols; i++) {
                    vec_length += u_vec[i] * u_vec[i];
                }
                u_length_squared = vec_length;
                vec_length = sqrt(vec_length + u_vec[col] * u_vec[col]);
                u_vec[col] -= vec_length;
                u_length_squared += u_vec[col] * u_vec[col];
                a_mat[col * num_cols + col] = vec_length;
            }
            else if(id > col) {
                a_mat[id * num_cols + col] = 0.0f;
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            
            /* Transform further columns of A */
            for(int i = col+1; i < num_cols; i++) {
                    if(id == 0) {
                        dot = 0.0f;
                        for(int j=col; j<num_cols; j++) {
                            dot += a_mat[j*num_cols + i] * u_vec[j];
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if(id >= col)
                    a_mat[id*num_cols + i] -= 2 * u_vec[id] *
                    dot / u_length_squared;
                    barrier(CLK_GLOBAL_MEM_FENCE);
                }
                if(id >= col) {
                    for(int i=col; i<num_cols; i++) {
                        p_mat[id*num_cols + i] = -2 * u_vec[i] *
                        u_vec[id] / u_length_squared;
                    }
                    p_mat[id*num_cols + id] += 1;
                }
                barrier(CLK_GLOBAL_MEM_FENCE);
     
                /* Multiply q_mat * p_mat = prod_mat */
                for(int i=col; i<num_cols; i++) {
                     prod = 0.0f;
                     for(int j=col; j<num_cols; j++) {
                             prod += q_mat[id*num_cols + j] *
                             p_mat[j*num_cols + i];
                        }
                     prod_mat[id*num_cols + i] = prod;
                }
                barrier(CLK_GLOBAL_MEM_FENCE);
     
         /* Place the content of prod_mat in q_mat */
         for(int i=col; i<num_cols; i++) {
                 q_mat[id*num_cols + i] =
                 prod_mat[id*num_cols + i];
                 }
         barrier(CLK_GLOBAL_MEM_FENCE);
         }
    }
    """).build()
     
     prod_mat = cl.Buffer(ctx, mf.WRITE_ONLY, prod_mat.nbytes)
     p_mat = cl.Buffer(ctx, mf.WRITE_ONLY, p_mat.nbytes)
     q_mat = cl.Buffer(ctx, mf.WRITE_ONLY, q_mat.nbytes)
     u_vec = cl.Buffer(ctx, mf.WRITE_ONLY, u_vec.nbytes)
     prg.qr(queue, prod_mat.shape, None, u_vec, a_mat, q_mat, p_mat, prod_mat)
     
     r = np.empty_like(prod_mat)
     cl.enqueue_copy(queue, r, prod_mat)
     
     root_sigmas = r[:, 0, 0].reshape(2207, 1)
     kalman_gains = r[:, 0, 1:] / root_sigmas
     updated_root_covs = r[:, 1:, 1:].transpose((0, 2, 1))
     updated_states = states + kalman_gains * residuals.reshape(2207, 1)
    
     
     return updated_states, updated_root_covs
