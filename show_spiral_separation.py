import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model("learn_spiral_separation.h5")

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)

X, Y = np.meshgrid(x, y)
X = X.flatten()
Y = Y.flatten()

I = model.predict(np.array([X, Y]).T)
I0 = I[:,0].reshape(120, 120)

plt.imshow(I0)
plt.show()