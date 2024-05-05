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

# Additional outputs and computations as needed, such as statistical tests and storing results in arrays
print("Simulations complete.")