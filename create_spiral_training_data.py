import numpy as np
import matplotlib.pyplot as plt

x_train = np.zeros((194,2))
y_train = np.zeros((194,1), dtype=np.uint8)

y_train[:97,0] = 0
y_train[97:,0] = 1

for i in range(97):
    r = 0.5 + i/16.0
    theta_black = -np.pi/2.0 + i*np.pi/16.0
    theta_white = np.pi/2.0 + i*np.pi/16.0
    x_black = r * np.cos(theta_black)
    y_black = r * np.sin(theta_black)
    x_white = r * np.cos(theta_white)
    y_white = r * np.sin(theta_white)
    x_train[i,0] = x_black
    x_train[i,1] = y_black
    x_train[97+i,0] = x_white
    x_train[97+i,1] = y_white

# plt.scatter(x_train[:97,0], x_train[:97,1])
# plt.scatter(x_train[97:,0], x_train[97:,1])
# plt.show()

np.savez("spiral_training_data.npz", x_train=x_train, y_train=y_train)