import numpy as np
from scipy import stats

X = np.array([-0.204681, -0.243221, -1.683012, 1.119622, 0.435571,
              1.025894, 0.84233, 0.522274, 0.147687, 0.040861])

Y = np.array([0.060793, 0.833231, -0.740263, 0.420588, 0.44041,
              -0.159056, 1.756728, 0.994368, -0.008375, -0.553091])

t_stat, p_val = stats.ttest_ind(X, Y)

print("alternative hypothesis: true difference in means is not equal to 0")
print(f"p-value: {p_val.round(2)}")
print(f"X Mean: {np.mean(X).round(2)}")
print(f"Y Mean: {np.mean(Y).round(2)}")