import numpy as np
from scipy.stats import f

# Daily return of SPY 09/03/24 - 09/17/24
X = np.array([-0.2, -0.24, -1.68, 1.12, 0.44, 1.03, 0.84, 0.52, 0.15, -0.01])
print("Variance of SPY:", np.var(X, ddof=1).round(2))

# Daily return of TSLA 09/03/24 - 09/17/24
Y = np.array([4.18, 4.90, -8.45, 2.63, 4.58, 0.87, 0.74, 0.21, -1.52, 0.25])
print("Variance of TSLA:", np.var(Y, ddof=1).round(2))

# F-test for equality of variances
f_stat = np.var(X, ddof=1) / np.var(Y, ddof=1)
dof1 = len(X) - 1  # degrees of freedom for X
dof2 = len(Y) - 1  # degrees of freedom for Y

p_value = 2 * min(f.cdf(f_stat, dof1, dof2), 1 - f.cdf(f_stat, dof1, dof2))

print("F-statistic:", f_stat.round(2))
print("p-value:", p_value.round(3))