import numpy as np
import matplotlib.pyplot as plt

# sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([3, 4, 2, 5, 6, 7, 8, 9, 10, 12])

n = len(x)

# calculate the means of x and y
mean_x = np.mean(x)
mean_y = np.mean(y)

# calculate slope and intercept
# slope (b1)
numerator = np.sum((x - mean_x) * (y - mean_y))
denominator = np.sum((x - mean_x) ** 2)
b1 = numerator / denominator

b0 = mean_y - (b1 * mean_x)

print(f"Coefficients:\nSlope (b1): {b1}\nIntercept (b0): {b0}")

# regression line
regression_line = b0 + b1 * x

# plot data points and regression line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Data Points', edgecolor='k', s=25)
plt.plot(x, regression_line, color='red', label='Regression Line', linewidth=1)
plt.title('Manual Linear Regression', fontsize=16)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gcf().canvas.manager.set_window_title("Linear Regression Plot")
plt.show()