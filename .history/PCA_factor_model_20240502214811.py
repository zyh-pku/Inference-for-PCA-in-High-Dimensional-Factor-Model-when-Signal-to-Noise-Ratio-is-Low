import numpy as np
import scipy.stats
from scipy.linalg import svd, diagsvd, block_diag
from numpy.linalg import inv

# Set seed for reproducibility
np.random.seed(123)

# Initial parameters
TT = 200  # length of time-series data
N = 300   # number of stocks
theta = 5.5  # SNR in L2 norm
n_trial = 200  # number of simulations

# Generate multivariate normal data
mu_B = np.array([0, 0, 0])
Sigma_B = np.diag([1, 1, 1])
B_0 = np.random.multivariate_normal(mu_B, Sigma_B, N)

# Adjustments for null hypothesis
B_0[:, 1] = B_0[:, 0]  # Copying first column to second

# Factor loadings under canonical condition
Sigma_f = np.diag([1, 1, 1]) * (theta / (3.091527 / np.sqrt(0.1)))**2
res_BF_svd = svd(B_0 @ np.sqrt(Sigma_f), full_matrices=False)
W = res_BF_svd[0]
Sig = diagsvd(res_BF_svd[1], len(res_BF_svd[1]), len(res_BF_svd[1]))
B = W @ Sig

# Generate noise covariance matrix
rho_max = 0.5
num_block = 20
n_sub = N // num_block
Sigma_u = None

for i in range(num_block):
    rho = np.random.uniform(0, rho_max)
    auto_corr_matrix = rho ** np.abs(np.arange(n_sub)[:, None] - np.arange(n_sub))
    if Sigma_u is None:
        Sigma_u = auto_corr_matrix
    else:
        Sigma_u = block_diag(Sigma_u, auto_corr_matrix)

# Estimation and thresholding of noise covariance matrix
C_thres = 1.0
thres = C_thres * (1 / np.sqrt(N) + np.sqrt(np.log(N) / TT))

# Placeholder for results arrays, similar to R's array initialization
# Initialize results arrays
B_CI_sum = np.zeros((N, n_trial))
F_CI_sum = np.zeros((TT, n_trial))
F_test_N_sum = np.zeros((4, n_trial))
F_test_D_sum = np.zeros((4, n_trial))
F_test_AM_sum = np.zeros((4, n_trial))
F_test_Bai_sum = np.zeros((4, n_trial))
B_rownorm_CI_sum = np.zeros((N, n_trial))
B_2row_CI_sum = np.zeros(n_trial)
B_2row_alter_CI_sum = np.zeros(n_trial)

# Simulations
for k in range(n_trial):
    print(k + 1)
    F_0 = np.random.multivariate_normal(mu_f, Sigma_f, TT)
    U_0 = np.random.multivariate_normal(np.zeros(N), Sigma_B, TT).T
    C_0 = B_0.dot(F_0.T)
    Y_0 = C_0 + U_0

    # Ground truth under canonical condition: factor realization
    F_1 = F_0 - np.ones(TT)[:, None] * F_0.mean(axis=0)
    FF = F_1.dot(inv(la.sqrtm(Sigma_f))).dot(res_BF_svd[2].T)

    # De-mean the data Y
    Y = Y_0 - Y_0.mean(axis=0) * np.ones(TT)[:, None].T
    res_svd = la.svd(Y, full_matrices=False)
    Sig_hat = np.diag(res_svd[1][:3]) / np.sqrt(TT)
    W_hat = res_svd[0]
    V_hat = res_svd[2].T

    B_hat = W_hat.dot(Sig_hat)
    F_hat = V_hat * np.sqrt(TT)

    # More computations needed for tests and estimations as per your R script
    # These would include PCA analysis, rotation matrix calculations, noise covariance estimations,
    # and other statistical measures you have outlined in the script.

    # Compute the SVD for groundtruth rotation matrix
    res_svd_ntl = svd(B.dot(FF.T))
    W_ntl = res_svd_ntl[0][:, :3]
    V_ntl = res_svd_ntl[2][:, :3].T
    Sig_ntl = diagsvd(res_svd_ntl[1][:3], 3, 3) / np.sqrt(TT)

    res_svd_W = svd(W_hat.T.dot(W_ntl))
    R_W = res_svd_W[0].dot(res_svd_W[2].T)

    res_svd_V = svd(V_hat.T.dot(V_ntl))
    R_V = res_svd_V[0].dot(res_svd_V[2].T)

    Q = W.T.dot(W_ntl)
    J = Sig.dot(Q).dot(inv(Sig_ntl))
    H = R_V.dot(J.T)

    # Estimate noise covariance matrix by hard-thresholding
    Sig_u_pivot = Y.dot(Y.T) / TT - B_hat.dot(B_hat.T)
    Corr_u_pivot = np.diag(np.sqrt(1 / np.diag(Sig_u_pivot))).dot(Sig_u_pivot).dot(np.diag(np.sqrt(1 / np.diag(Sig_u_pivot))))
    idx = np.abs(Corr_u_pivot) > thres
    tmp = np.diag(np.ones(N))
    tmp[idx] = Corr_u_pivot[idx]
    Sig_u_hat = np.diag(np.sqrt(np.diag(Sig_u_pivot))).dot(tmp).dot(np.diag(np.sqrt(np.diag(Sig_u_pivot))))

    # Groundtruth of factor loading after rotation
    B_gt = B.dot(inv(H))
    Chi_B = (1 / ((1 / TT) * np.diag(Sig_u_hat))) * np.diag((B_hat - B_gt).dot((B_hat - B_gt).T))

    # Groundtruth of factor realization after rotation
    F_gt = FF.dot(H.T)
    Q_V = inv(Sig_hat).dot(W_hat.T).dot(np.diag(np.diag(Sig_u_hat))).dot(W_hat).dot(inv(Sig_hat))
    Chi_F = np.diag((F_hat - F_gt).dot(inv(Q_V)).dot((F_hat - F_gt).T))

    B_CI_sum[:, k] = Chi_B
    F_CI_sum[:, k] = Chi_F

    # Row norm computations
    B_ntl = W_ntl.dot(Sig_ntl)
    B_ntl_row_norm_square = np.diag(B_ntl.dot(B_ntl.T))
    B_hat_row_norm_square = np.diag(B_hat.dot(B_hat.T))
    B_row_norm_square = np.diag(B.dot(B.T))
    B_rownorm_CI_sum[:, k] = (B_row_norm_square - B_hat_row_norm_square) / (2 / np.sqrt(TT) * np.sqrt(np.diag(Sig_u_hat)) * np.sqrt(B_hat_row_norm_square))

    # Test two rows of B -- under Null and Alternative
    B_2row_CI_sum[k] = np.sum((B_hat[B_index_1, :] - B_hat[B_index_2, :]) ** 2) * TT / (Sig_u_hat[B_index_1, B_index_1] + Sig_u_hat[B_index_2, B_index_2] - 2 * Sig_u_hat[B_index_1, B_index_2])
    B_2row_alter_CI_sum[k] = np.sum((B_hat[B_index_1, :] - B_hat[B_index_3, :]) ** 2) * TT / (Sig_u_hat[B_index_1, B_index_1] + Sig_u_hat[B_index_3, B_index_3] - 2 * Sig_u_hat[B_index_1, B_index_3])

    # Tests the factor -- code here will include the calculations for Test.N and Test.D, using the methodology outlined
    ## Whole period
    t_set = np.arange(TT)  # full time period
    w = np.array([1, 1, 0.5])
    V_pinv = inv(V_hat[t_set, :].T @ V_hat[t_set, :]) @ V_hat[t_set, :].T
    Q_V = inv(Sig_hat) @ W_hat.T @ Sig_u_hat @ W_hat @ inv(Sig_hat)
    
    NT_sub = int(np.floor(np.sqrt(min(N, TT))))
    Q_V_CSHAC = (N / NT_sub) * inv(Sig_hat) @ W_hat[:NT_sub, :].T @ Sig_u_pivot[:NT_sub, :NT_sub] @ W_hat[:NT_sub, :] @ inv(Sig_hat)
    
    # Null scenario
    v_set = FF[t_set, :] @ w
    Test_D = v_set.T @ V_pinv @ Q_V @ V_pinv @ v_set
    Test_N = v_set.T @ v_set - v_set.T @ V_hat[t_set, :] @ V_pinv @ v_set
    F_test_N_sum[0, k] = Test_N
    F_test_D_sum[0, k] = Test_D
    
    G_set = v_set - V_hat[t_set, :] @ V_pinv @ v_set
    F_test_D_Bai = v_set.T @ V_pinv @ Q_V_CSHAC @ V_pinv @ v_set
    tao_set = G_set / np.sqrt(F_test_D_Bai[0, 0])
    F_test_Bai_sum[0, k] = np.sum(np.abs(tao_set) > norm.ppf(0.975)) / len(tao_set)
    F_test_Bai_sum[1, k] = np.max(np.abs(tao_set))
    
    # Alternative scenario
    cmp = (np.eye(len(t_set)) - FF[t_set, :] @ inv(FF[t_set, :].T @ FF[t_set, :]) @ FF[t_set, :].T) @ np.random.randn(len(t_set))
    v_set = FF[t_set, :] @ w + cmp / np.linalg.norm(cmp, 2) * np.linalg.norm(FF[t_set, :], 'fro') * np.sqrt(np.sum(w**2))
    Test_D = v_set.T @ V_pinv @ Q_V @ V_pinv @ v_set
    Test_N = v_set.T @ v_set - v_set.T @ V_hat[t_set, :] @ V_pinv @ v_set
    F_test_N_sum[1, k] = Test_N
    F_test_D_sum[1, k] = Test_D
    
    G_set = v_set - V_hat[t_set, :] @ V_pinv @ v_set
    F_test_D_Bai = v_set.T @ V_pinv @ Q_V_CSHAC @ V_pinv @ v_set
    tao_set = G_set / np.sqrt(F_test_D_Bai[0, 0])
    F_test_Bai_sum[2, k] = np.sum(np.abs(tao_set) > norm.ppf(0.975)) / len(tao_set)
    F_test_Bai_sum[3, k] = np.max(np.abs(tao_set))
    
    # Subset time period
    t_set = np.arange(100, 112)  # subset period
    V_pinv = inv(V_hat[t_set, :].T @ V_hat[t_set, :]) @ V_hat[t_set, :].T
    Q_V = inv(Sig_hat) @ W_hat.T @ Sig_u_hat @ W_hat @ inv(Sig_hat)
    
    # Null scenario
    v_set = FF[t_set, :] @ w
    Test_D = v_set.T @ V_pinv @ Q_V @ V_pinv @ v_set
    Test_N = v_set.T @ v_set - v_set.T @ V_hat[t_set, :] @ V_pinv @ v_set
    F_test_N_sum[2, k] = Test_N
    F_test_D_sum[2, k] = Test_D
    
    G_set = v_set - V_hat[t_set, :] @ V_pinv @ v_set
    tao_set = G_set / np.sqrt(Test_D)
    F_test_AM_sum[0, k] = np.sum(np.abs(tao_set) > norm.ppf(0.975)) / len(tao_set)
    F_test_AM_sum[1, k] = np.max(np.abs(tao_set))
    
    # Alternative scenario for the subset period
    cmp = (np.eye(len(t_set)) - FF[t_set, :] @ inv(FF[t_set, :].T @ FF[t_set, :]) @ FF[t_set, :].T) @ np.random.randn(len(t_set))
    v_set = FF[t_set, :] @ w + 2 * cmp / np.linalg.norm(cmp, 2) * np.linalg.norm(FF[t_set, :], 'fro') * np.sqrt(np.sum(w**2))
    Test_D = v_set.T @ V_pinv @ Q_V @ V_pinv @ v_set
    Test_N = v_set.T @ v_set - v_set.T @ V_hat[t_set, :] @ V_pinv @ v_set
    F_test_N_sum[3, k] = Test_N
    F_test_D_sum[3, k] = Test_D
    
    G_set = v_set - V_hat[t_set, :] @ V_pinv @ v_set
    tao_set = G_set / np.sqrt(Test_D)
    F_test_AM_sum[2, k] = np.sum(np.abs(tao_set) > norm.ppf(0.975)) / len(tao_set)
    F_test_AM_sum[3, k] = np.max(np.abs(tao_set))

# Additional outputs and computations as needed, such as statistical tests and storing results in arrays
print("Simulations complete.")