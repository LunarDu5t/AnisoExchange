import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

ax = plt.figure().add_subplot(projection='3d')
vectors = np.array([[2.0, 1.0, 0.4],
                    [0.5, 2.0, 0.2],
                    [0.5, 0.4, 1.0]])

def orthogonalize(vecs):
    overlap = vecs.T@vecs
    s_inv = sp.linalg.fractional_matrix_power(overlap, -0.5)
    nvecs = vecs@s_inv
    return nvecs

def plot_vecs(vecs, col=["red", "green", "blue"], lw=1.0):
    rows, cols = vecs.shape
    for c in range(cols):
        ax.plot([0, vecs[0, c]], [0, vecs[1, c]], [0, vecs[2, c]], color=col[c], linewidth=lw)

plot_vecs(vectors)
nvectors = orthogonalize(vectors)
plot_vecs(nvectors, col=["#FF7F7F", "#90EE90", "#ADD8E6"], lw=5.0)

print("Overlap of initvecs")
print(vectors.T@vectors)
print("Overlap of new vectors")
print(nvectors.T@nvectors)
plt.show()