def func(x):
    x1, x2 = x
    return 1 / x1 ** 2 + x1 ** 2 + 1 / x2 ** 2 + x2 ** 2


from AFSA import AFSA

afsa = AFSA(func, n_dim=2, size_pop=50, max_iter=300,
            max_try_num=100, step=0.5, visual=0.3,
            q=0.98, delta=0.5)
best_x, best_y = afsa.run()
print(best_x, best_y)

import matplotlib.pyplot as plt

# First iteration fish positions
positions_start = afsa.history_positions[0]

# Last iteration fish positions
positions_end = afsa.history_positions[-1]

# Plot positions
plt.figure(figsize=(12, 6))

# First iteration
plt.subplot(1, 2, 1)
plt.scatter(positions_start[:, 0], positions_start[:, 1], c='blue', alpha=0.7)
plt.title('Fish Positions - Iteration 1')
plt.xlabel('X1')
plt.ylabel('X2')

# Last iteration
plt.subplot(1, 2, 2)
plt.scatter(positions_end[:, 0], positions_end[:, 1], c='red', alpha=0.7)
plt.title('Fish Positions - Final Iteration')
plt.xlabel('X1')
plt.ylabel('X2')

plt.tight_layout()
plt.show()

# Now creating a new figure for both iterations combined
plt.figure(figsize=(6, 6))

# Scatter plot showing both first and last iteration positions
plt.scatter(positions_start[:, 0], positions_start[:, 1], c='blue', label='First Iteration', alpha=0.7)
plt.scatter(positions_end[:, 0], positions_end[:, 1], c='red', label='Final Iteration', alpha=0.7)

# Adding title and labels
plt.title('Fish Positions - First and Final Iteration Combined')
plt.xlabel('X1')
plt.ylabel('X2')

# Show legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()