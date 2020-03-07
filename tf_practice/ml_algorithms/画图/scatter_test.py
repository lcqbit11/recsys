import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


N = 100
r0 = 0.6
x = 0.9 * np.random.rand(N)
y = 0.9 * np.random.rand(N)
print(x)
print(y)
color = []
for i in range(N):
    color.append(i)

plt.scatter(x, y, marker='o', c=color)

# area = (20 * np.random.rand(N))**2  # 0 to 10 point radii
# c = np.sqrt(area)
#
# plt.scatter(x, y, marker='^', c=c)


plt.show()