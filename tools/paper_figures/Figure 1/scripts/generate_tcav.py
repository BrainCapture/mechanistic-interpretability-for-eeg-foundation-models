import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

np.random.seed(42)

# Generate Class A (Concept C)
mA = [2, 3.5]
covA = [[1.5, 0.5], [0.5, 1.2]]
A = np.random.multivariate_normal(mA, covA, 15)

# Generate Class B (Not C)
mB = [6, 1.5]
covB = [[2.0, -0.2], [-0.2, 1.8]]
B = np.random.multivariate_normal(mB, covB, 15)

# Combine and fit LDA
X = np.vstack((A, B))
y = np.hstack((np.ones(A.shape[0]), np.zeros(B.shape[0])))

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

# Boundary w^T x + b = 0
w = lda.coef_[0]
b = lda.intercept_[0]

v_C = w / np.linalg.norm(w)
fig, ax = plt.subplots(figsize=(2.8, 1.2))
color = '#E08A1F'
cav_color = 'black'

# Plot A (Concept C)
ax.scatter(A[:, 0], A[:, 1], c=color, s=50, alpha=0.85, zorder=2, edgecolors='none')
# Plot B (Not C)
ax.scatter(B[:, 0], B[:, 1], c='white', edgecolors=color, linewidths=1.5, s=50, alpha=0.9, zorder=2)

xlim = ax.get_xlim()
xx = np.linspace(xlim[0] - 1, xlim[1] + 1, 100)
yy = -(w[0]*xx + b) / w[1]

ax.plot(xx, yy, color=color, linewidth=1.6, linestyle='--', zorder=1)

# Normal vector (CAV)
mid_x = (np.mean(A[:,0]) + np.mean(B[:,0])) / 2
mid_y = -(w[0]*mid_x + b) / w[1]
arrow_len = 1.0
ax.arrow(mid_x, mid_y, v_C[0]*arrow_len, v_C[1]*arrow_len,
         head_width=0.25, head_length=0.3, fc=cav_color, ec=cav_color, linewidth=1.5, zorder=4)

# We omit the text label this time as requested, since it overlaps.

# Aspect equal to make orthogonal show as orthogonal visually
ax.set_aspect('equal')
ax.axis('off')

# Limits
ax.set_xlim(min(A[:,0].min(), B[:,0].min()) - 0.5, max(A[:,0].max(), B[:,0].max()) + 0.5)
ax.set_ylim(min(A[:,1].min(), B[:,1].min()) - 0.5, max(A[:,1].max(), B[:,1].max()) + 0.5)

plt.tight_layout(pad=0)
fig.savefig('tcav.svg', transparent=True, bbox_inches='tight', pad_inches=0.01)
plt.close(fig)
