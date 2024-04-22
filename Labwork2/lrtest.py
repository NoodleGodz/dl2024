import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 100

# Generate random values for features x1 and x2
x1 = np.random.uniform(low=-10, high=10, size=num_samples)
x2 = np.random.uniform(low=-10, high=10, size=num_samples)

# Generate random binary labels (0 or 1)
y = np.random.randint(2, size=num_samples)

# Display the first few samples to verify
print("x1:", x1[:5])
print("x2:", x2[:5])
print("y:", y[:5])
