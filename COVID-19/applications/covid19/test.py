import autograd.numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-3, 3, 200)
y = (np.tanh(x)+1)/2

plt.figure()
plt.plot(x,y,'.-')
plt.show()

