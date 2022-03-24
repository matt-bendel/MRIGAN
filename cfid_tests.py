# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np
import tensorflow as tf

# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt
def sqrt_newton_schulz(A, numIters, dtype=None):
    with torch.no_grad():
        if dtype is None:
            dtype = A.type()
        batchSize = A.shape[0]
        dim = A.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A)).to(A.device)
        I = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype).to(A.device)
        Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype).to(A.device)
        for i in range(numIters):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
        sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA

def symmetric_matrix_square_root(mat, eps=1e-10):
    """Compute square root of a symmetric matrix.
    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat.
    Also note that this method **only** works for symmetric matrices.
    Args:
      mat: Matrix to take the square root of.
      eps: Small epsilon such that any element less than eps will not be square
        rooted to guard against numerical instability.
    Returns:
      Matrix square root of mat.
    """
    # Unlike numpy, tensorflow's return order is (s, u, v)
    s, u, v = tf.linalg.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = tf.compat.v1.where(tf.less(s, eps), s, tf.sqrt(s))
    # Note that the v returned by Tensorflow is v = V
    # (when referencing the equation A = U S V^T)
    # This is unlike Numpy which returns v = V^T
    return tf.matmul(tf.matmul(u, tf.linalg.tensor_diag(si)), v, transpose_b=True)


def symmetric_matrix_square_root_torch(mat, eps=1e-10):
    """Compute square root of a symmetric matrix.
    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat.
    Also note that this method **only** works for symmetric matrices.
    Args:
      mat: Matrix to take the square root of.
      eps: Small epsilon such that any element less than eps will not be square
        rooted to guard against numerical instability.
    Returns:
      Matrix square root of mat.
    """
    # Unlike numpy, tensorflow's return order is (s, u, v)
    u, s, v = torch.linalg.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = s
    si[torch.where(si >= eps)] = torch.sqrt(si[torch.where(si >= eps)])

    # Note that the v returned by Tensorflow is v = V
    # (when referencing the equation A = U S V^T)
    # This is unlike Numpy which returns v = V^T
    return torch.matmul(torch.matmul(u, torch.diag(si)), v)


def trace_sqrt_product(sigma, sigma_v):
    """Find the trace of the positive sqrt of product of covariance matrices.
    '_symmetric_matrix_square_root' only works for symmetric matrices, so we
    cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
    ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).
    Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
    We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
    Note the following properties:
    (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
      => eigenvalues(A A B B) = eigenvalues (A B B A)
    (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
      => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
    (iii) forall M: trace(M) = sum(eigenvalues(M))
      => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                    = sum(sqrt(eigenvalues(A B B A)))
                                    = sum(eigenvalues(sqrt(A B B A)))
                                    = trace(sqrt(A B B A))
                                    = trace(sqrt(A sigma_v A))
    A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
    use the _symmetric_matrix_square_root function to find the roots of these
    matrices.
    Args:
      sigma: a square, symmetric, real, positive semi-definite covariance matrix
      sigma_v: same as sigma
    Returns:
      The trace of the positive square root of sigma*sigma_v
    """

    # Note sqrt_sigma is called "A" in the proof above
    sqrt_sigma = symmetric_matrix_square_root(sigma)

    # This is sqrt(A sigma_v A) above
    sqrt_a_sigmav_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_v, sqrt_sigma))

    return tf.linalg.trace(symmetric_matrix_square_root(sqrt_a_sigmav_a))


def trace_sqrt_product_torch(sigma, sigma_v):
    """Find the trace of the positive sqrt of product of covariance matrices.
    '_symmetric_matrix_square_root' only works for symmetric matrices, so we
    cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
    ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).
    Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
    We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
    Note the following properties:
    (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
      => eigenvalues(A A B B) = eigenvalues (A B B A)
    (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
      => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
    (iii) forall M: trace(M) = sum(eigenvalues(M))
      => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                    = sum(sqrt(eigenvalues(A B B A)))
                                    = sum(eigenvalues(sqrt(A B B A)))
                                    = trace(sqrt(A B B A))
                                    = trace(sqrt(A sigma_v A))
    A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
    use the _symmetric_matrix_square_root function to find the roots of these
    matrices.
    Args:
      sigma: a square, symmetric, real, positive semi-definite covariance matrix
      sigma_v: same as sigma
    Returns:
      The trace of the positive square root of sigma*sigma_v
    """

    # Note sqrt_sigma is called "A" in the proof above
    sqrt_sigma = symmetric_matrix_square_root_torch(sigma)

    # This is sqrt(A sigma_v A) above
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))

    return torch.trace(symmetric_matrix_square_root_torch(sqrt_a_sigmav_a))


# **Estimators**
#
def sample_covariance(a, b, invert=False):
    '''
    Sample covariance estimating
    a = [N,m]
    b = [N,m]
    '''
    assert (a.shape[0] == b.shape[0])
    assert (a.shape[1] == b.shape[1])
    m = a.shape[1]
    N = a.shape[0]
    C = tf.matmul(tf.transpose(a), b) / N
    if invert:
        return tf.linalg.pinv(C)
    else:
        return C


def sample_covariance_torch(a, b):
    '''
    Sample covariance estimating
    a = [N,m]
    b = [N,m]
    '''
    assert (a.shape[0] == b.shape[0])
    assert (a.shape[1] == b.shape[1])
    m = a.shape[1]
    N = a.shape[0]
    return torch.matmul(torch.transpose(a, 0, 1), b) / N

def get_cfid_torch_svd(y_predict, x_true, y_true, np_inv=False, mat=False):
    # mean estimations
    y_true = y_true.to(x_true.device)
    m_y_predict = torch.mean(y_predict, dim=0)
    m_x_true = torch.mean(x_true, dim=0)
    m_y_true = torch.mean(y_true, dim=0)

    no_m_y_true = y_true - m_y_true
    no_m_y_pred = y_predict - m_y_predict
    no_m_x_true = x_true - m_x_true

    m_dist = torch.einsum('...k,...k->...', m_y_true - m_y_predict, m_y_true - m_y_predict)

    u, s, vh = torch.linalg.svd(no_m_x_true, full_matrices=False)
    v = vh.t()
    c_dist_1 = torch.norm(torch.matmul(no_m_y_true.t() - no_m_y_pred.t(), v)) ** 2 / y_true.shape[0]

    v_t_v = torch.matmul(vh, v)
    y_pred_w_v_t_v = torch.matmul(no_m_y_pred.t(), torch.matmul(v_t_v, no_m_y_pred))
    y_true_w_v_t_v = torch.matmul(no_m_y_true.t(), torch.matmul(v_t_v, no_m_y_true))

    c_y_true_given_x_true = 1 / y_true.shape[0] * (torch.matmul(no_m_y_true.t(), no_m_y_true) - y_true_w_v_t_v)
    c_y_predict_given_x_true = 1 / y_true.shape[0] * (torch.matmul(no_m_y_pred.t(), no_m_y_pred) - y_pred_w_v_t_v)

    c_dist_2 = torch.trace(c_y_true_given_x_true + c_y_predict_given_x_true) - 2 * trace_sqrt_product_torch(
        c_y_predict_given_x_true, c_y_true_given_x_true)

    return m_dist + c_dist_1 + c_dist_2, c_dist_1.cpu().numpy(), c_dist_2.cpu().numpy()

def get_cfid_torch(y_predict, x_true, y_true, np_inv=False, mat=False):
    # mean estimations
    m_y_predict = torch.mean(y_predict, dim=0)
    m_x_true = torch.mean(x_true, dim=0)

    print('SHAPE X: ', x_true.shape)
    print('RANK X: ', torch.matrix_rank(x_true))

    # covariance computations
    c_y_predict_x_true = sample_covariance_torch(y_predict - m_y_predict, x_true - m_x_true)
    c_y_predict_y_predict = sample_covariance_torch(y_predict - m_y_predict, y_predict - m_y_predict)
    c_x_true_y_predict = sample_covariance_torch(x_true - m_x_true, y_predict - m_y_predict)

    y_true = y_true.to(x_true.device)
    m_y_true = torch.mean(y_true, dim=0)
    c_y_true_x_true = sample_covariance_torch(y_true - m_y_true, x_true - m_x_true)
    c_x_true_y_true = sample_covariance_torch(x_true - m_x_true, y_true - m_y_true)
    c_y_true_y_true = sample_covariance_torch(y_true - m_y_true, y_true - m_y_true)

    no_m_y_true = y_true - m_y_true
    n_m_y_pred = y_predict - m_y_predict
    n_m_x_true = x_true - m_x_true

    other_temp = torch.norm((no_m_y_true.t() - n_m_y_pred.t()))**2 / y_true.shape[0]
    u, s, vh = torch.linalg.svd(no_m_x_true, full_matrices=False)
    v = vh.t()
    #TODO: REMOVE TRANSPOSE IN DATA RICH SCENARIO
    svd_temp = torch.norm(torch.matmul(no_m_y_true.t() - n_m_y_pred.t(), v))**2 / y_true.shape[0]

    x_t_x = sample_covariance_torch(x_true - m_x_true, x_true - m_x_true)
    dtype = np.float32
    inv_c_x_true_x_true = torch.tensor(np.linalg.pinv(x_t_x.cpu().numpy(), rcond=10.*2048*np.finfo(dtype).eps)).to(y_true.device) if np_inv else torch.linalg.pinv(x_t_x, rcond=10.*2048*np.finfo(dtype).eps)

    if mat:
        return c_y_predict_x_true.cpu().numpy(), c_y_predict_y_predict.cpu().numpy(), c_x_true_y_predict.cpu().numpy(), c_y_true_x_true.cpu().numpy(), c_x_true_y_true.cpu().numpy(), c_y_true_y_true.cpu().numpy()

    c_y_true_given_x_true = c_y_true_y_true - torch.matmul(c_y_true_x_true,
                                                           torch.matmul(inv_c_x_true_x_true, c_x_true_y_true))

    c_y_predict_given_x_true = c_y_predict_y_predict - torch.matmul(c_y_predict_x_true,
                                                                    torch.matmul(inv_c_x_true_x_true,
                                                                                 c_x_true_y_predict))

    c_y_true_x_true_minus_c_y_predict_x_true = c_y_true_x_true - c_y_predict_x_true
    c_x_true_y_true_minus_c_x_true_y_predict = c_x_true_y_true - c_x_true_y_predict

    m_dist = torch.einsum('...k,...k->...', m_y_true - m_y_predict, m_y_true - m_y_predict)
    c_dist1 = torch.trace(torch.matmul(torch.matmul(c_y_true_x_true_minus_c_y_predict_x_true, inv_c_x_true_x_true),
                                       c_x_true_y_true_minus_c_x_true_y_predict))
    c_dist2 = torch.trace(c_y_true_given_x_true + c_y_predict_given_x_true) - 2 * trace_sqrt_product_torch(
        c_y_predict_given_x_true, c_y_true_given_x_true)

    return m_dist + c_dist1 + c_dist2, c_dist1.cpu().numpy(), other_temp.cpu().numpy(), c_dist2.cpu().numpy(), inv_c_x_true_x_true.cpu().numpy(), svd_temp


def get_cfid(y_predict, x_true, y_true, np_inv=False, torch_inv_matrix=None, mat=False):
    assert ((y_predict.shape[0] == y_true.shape[0]) and (y_predict.shape[0] == x_true.shape[0]))
    assert ((y_predict.shape[1] == y_true.shape[1]) and (y_predict.shape[1] == x_true.shape[1]))

    # mean estimations
    m_y_predict = tf.reduce_mean(y_predict, axis=0)
    m_x_true = tf.reduce_mean(x_true, axis=0)

    # covariance computations
    c_y_predict_x_true = sample_covariance(y_predict - m_y_predict, x_true - m_x_true)
    c_y_predict_y_predict = sample_covariance(y_predict - m_y_predict, y_predict - m_y_predict)
    c_x_true_y_predict = sample_covariance(x_true - m_x_true, y_predict - m_y_predict)

    y_true = tf.convert_to_tensor(y_true)
    m_y_true = tf.reduce_mean(y_true, axis=0)
    c_y_true_x_true = sample_covariance(y_true - m_y_true, x_true - m_x_true)
    c_x_true_y_true = sample_covariance(x_true - m_x_true, y_true - m_y_true)
    c_y_true_y_true = sample_covariance(y_true - m_y_true, y_true - m_y_true)

    x_t_x = sample_covariance(x_true - m_x_true, x_true - m_x_true)
    dtype = np.float32
    inv_c_x_true_x_true = tf.convert_to_tensor(np.linalg.pinv(x_t_x.numpy(), rcond=10.*2048*np.finfo(dtype).eps)) if np_inv else sample_covariance(x_true - m_x_true, x_true - m_x_true, invert=True)

    # inv_c_x_true_x_true = tf.convert_to_tensor(torch_inv_matrix)
    if mat:
        return c_y_predict_x_true.numpy(), c_y_predict_y_predict.numpy(), c_x_true_y_predict.numpy(), c_y_true_x_true.numpy(), c_x_true_y_true.numpy(), c_y_true_y_true.numpy()

    # conditoinal mean and covariance estimations
    v = x_true - m_x_true

    A = tf.matmul(inv_c_x_true_x_true, tf.transpose(v))

    c_y_true_given_x_true = c_y_true_y_true - tf.matmul(c_y_true_x_true,
                                                        tf.matmul(inv_c_x_true_x_true, c_x_true_y_true))
    c_y_predict_given_x_true = c_y_predict_y_predict - tf.matmul(c_y_predict_x_true,
                                                                 tf.matmul(inv_c_x_true_x_true, c_x_true_y_predict))
    c_y_true_x_true_minus_c_y_predict_x_true = c_y_true_x_true - c_y_predict_x_true
    c_x_true_y_true_minus_c_x_true_y_predict = c_x_true_y_true - c_x_true_y_predict

    # Distance between Gaussians
    m_dist = tf.einsum('...k,...k->...', m_y_true - m_y_predict, m_y_true - m_y_predict)
    c_dist1 = tf.linalg.trace(tf.matmul(tf.matmul(c_y_true_x_true_minus_c_y_predict_x_true, inv_c_x_true_x_true),
                                        c_x_true_y_true_minus_c_x_true_y_predict))
    c_dist2 = tf.linalg.trace(c_y_true_given_x_true + c_y_predict_given_x_true) - 2 * trace_sqrt_product(
        c_y_predict_given_x_true, c_y_true_given_x_true)

    return m_dist + c_dist1 + c_dist2, c_dist1.numpy(), c_dist2.numpy()


# A pytorch implementation of cov, from Modar M. Alfadly
# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
                m: A 1-D or 2-D array containing multiple variables and observations.
                        Each row of `m` represents a variable, and each column a single
                        observation of all those variables.
                rowvar: If `rowvar` is True, then each row represents a
                        variable, with observations in the columns. Otherwise, the
                        relationship is transposed: each column represents a variable,
                        while the rows contain observations.

        Returns:
                The covariance matrix of the variables.
        '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


if __name__ == '__main__':
    mat_test = False
    recon_embeds = torch.load('image_embeds_7000.pt').to(dtype=torch.float64)
    cond_embeds = torch.load('cond_embeds_7000.pt').to(dtype=torch.float64)
    gt_embeds = torch.load('true_embeds_7000.pt').to(dtype=torch.float64)

    if mat_test:
        m1_1, m2_1, m3_1, m4_1, m5_1, m6_1 = get_cfid_torch(recon_embeds, cond_embeds, gt_embeds, mat=mat_test)
        # with tf.device('/gpu:3'):
        #     m1_2, m2_2, m3_2, m4_2, m5_2, m6_2 = get_cfid(tf.convert_to_tensor(recon_embeds.cpu().numpy()),
        #                                                     tf.convert_to_tensor(cond_embeds.cpu().numpy()),
        #                                                     gt_embeds.cpu().numpy(), mat=mat_test)
        # print(np.linalg.norm(m1_1 - m1_2))
        # print(np.linalg.norm(m2_1 - m2_2))
        # print(np.linalg.norm(m3_1 - m3_2))
        # print(np.linalg.norm(m4_1 - m4_2))
        # print(np.linalg.norm(m5_1 - m5_2))
        # print(np.linalg.norm(m6_1 - m6_2))

        exit()

    cfid1, c_dist_torch, c_dist_fro_norm, c_dist_2_pt, torch_mat, c_dist_svd = get_cfid_torch(recon_embeds, cond_embeds, gt_embeds)
    cfid_svd, cdist1_svd, cdist2 = get_cfid_torch_svd(recon_embeds, cond_embeds, gt_embeds)
    # cfid_np, c_dist_np, _, c_dist_2_np, _, _ = get_cfid_torch(recon_embeds, cond_embeds, gt_embeds, np_inv=True)
    # with tf.device('/gpu:3'):
    #     cfid2, c_dist_tf, c_dist_2_tf = get_cfid(tf.convert_to_tensor(recon_embeds.cpu().numpy()),
    #                               tf.convert_to_tensor(cond_embeds.cpu().numpy()), gt_embeds.cpu().numpy(), torch_inv_matrix=torch_mat)

    print('CFID TORCH: ', cfid1.cpu().numpy())
    print('CFID TORCH SVD: ', cfid_svd.cpu().numpy())
    # print('CFID TF: ', cfid2.numpy())

    print('\n')

    print('CDIST_1 TORCH: ', c_dist_torch)
    print('CDIST_1 TORCH SVD: ', cdist1_svd)
    # print('CDIST_1 NP: ', c_dist_np)
    print('CDIST_1 FRO NORM: ', c_dist_fro_norm)
    print('CDIST_1 SVD: ', c_dist_svd)

    print('\n')

    print('CDIST_2 TORCH: ', c_dist_2_pt)
    print('CDIST_2 TORCH SVD: ', cdist2_svd)
    # print('CDIST_2 TF: ', c_dist_2_tf)
    # print('CDIST_2 NP: ', c_dist_2_np)

    print('\n')
