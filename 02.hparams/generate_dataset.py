
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)

# true polynomial
poly = np.polynomial.Polynomial([7.2, -7.2, -0.2, 1])

# generate domain and value
x1 = np.concat(( # domain of train set
    np.linspace(-4.0, -3.1, 10), np.linspace(-0.2, 0.2, 5),
    np.linspace(3.1, 4.0, 10)
))
x1 = x1 + np.random.uniform(-0.1, 0.1, size=x1.shape)
x2 = np.concat(( # domain of test set
    np.linspace(-2.0, -1.0, 4), np.linspace(1.0, 2.0, 4),
    np.linspace(-5.0, -4.0, 4), np.linspace(4.0, 5.0, 4)
))
x2 = x2 + np.random.uniform(-0.05, 0.05, size=x2.shape)

y1 = poly(x1) + np.random.uniform(-1.5, 1.5, size=x1.shape)
y2 = poly(x2) + np.random.uniform(-0.5, 0.5, size=x2.shape)

# save as numpy npz
train_dir = Path('datasets/train/')
if not train_dir.exists():
    train_dir.mkdir(parents=True)
np.savez(train_dir / 'data.npz', x=x1, y=y1)
test_dir = Path('datasets/test/')
if not test_dir.exists():
    test_dir.mkdir(parents=True)
np.savez(test_dir / 'data.npz', x=x2, y=y2)

# plot
plt.plot(x1, y1, 'o', label='train')
plt.plot(x2, y2, 'o', label='test')
plt.legend()
plt.savefig('dataset.png')
