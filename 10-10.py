import numpy as np
import matplotlib.pyplot as plt

def sigmoide(x):
    return 1/(1+np.exp(-x))

x1 = np.linspace(-100, 100, 1000)
plt.plot(x1, sigmoide(x1))
plt.show()
#Esse plot parece a função degrau, mas não é
#Isso é pq os valores de X são muito grandes
#Então é necessário que sejam feitas operações onde os dados não perdem sua propocionalidade,
#mas que ficam menores