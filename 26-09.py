import numpy as np
import matplotlib.pyplot as plt

x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
])

plt.scatter(x[0,0], x[0,1], color="green", edgecolors='k', linewidth=2)
plt.scatter(x[1:,0], x[1:,1], color="purple", edgecolors='k', linewidth=2)
plt.xlim(-.1,1.1)
plt.ylim(-.1,1.1)

# u = w1x1 + w2x2 - theta, verificar os slides no problema OR
'''
w1x1 + w2x2 = theta
w1x1/w2 + x2 = theta/w2
x2 = theta/w2 - w1x1/w2
'''
theta = 1.5
w1 = 1.7
w2 = 2.8

# Gerando dados abaixo para fazer um plot
x1 = np.linspace(-2,2,20)
#x2 = theta/w2 - w1/w2*x1
x2 = theta/w2 - w1*x1/w2

plt.plot(x1, x2, color="red")
plt.show()

'''
Aqui temos um neurônio, mas ele não é dotado de inteligência, pois não tem uma regra
de aprendizado, os valores de theta, w1 e w2 foram estipulados por nós.
'''