import numpy as np
import matplotlib.pyplot as plt

def plotar():
    plt.clf()
    plt.ylim(-10, 5)
    plt.scatter(Data[:, :2][0:1500,0],Data[:, :2][0:1500,1],color='blue',edgecolors='k')
    plt.scatter(Data[:, :2][1500:,0],Data[:, :2][1500:,1],color='red',edgecolors='k')
    plt.plot(x1, x2, color = 'k', linewidth=3) 
    plt.show()

def sinal(u):
    if u>=0:
        return 1
    else:
        return -1

# COLETA E TRATAMENTO DE DADOS
Data = np.loadtxt("./DataAV2.csv", delimiter=',')
x = Data[:, 0:2]
x = x.T
p,n = x.shape
x = np.concatenate((
    -np.ones((1,n)),x
))
y = Data[:, 2].reshape((n, 1))
print(f"Shape de x: {x.shape}") # p+1 x n
print(f"Shape de y: {y.shape}") # n x 1

def embaralhar_dados():
    seed = np.random.permutation(n)
    x_random = x[:, seed]
    y_random = y[seed, :]

    x_treino = x_random[:, 0 : int(n * 0.8)]
    y_treino = y_random[0 : int(n * 0.8), :]

    x_teste = x_random[:, int(n * 0.8):]
    y_teste = y_random[int(n * 0.8):, :]

    return (x_treino, y_treino, x_teste, y_teste)

rodadas = 100
for r in range(rodadas):
    (x_treino, y_treino, x_teste, y_teste) = embaralhar_dados()
    
    #w = np.zeros((p+1, 1))
    w = np.random.rand(p+1, 1)
    x1 = np.linspace(-12, 12, n)
    x2 = np.zeros((n,))
    LR = 0.01

    erro = True
    epoch = 0
    while(erro and epoch < 100):
        erro = False
        epoch = epoch + 1
        #print(f"\rÉpoca: {epoch}", end="")
        for t in range(x_treino.shape[1]):
            x_t = x_treino[:,t].reshape((p+1,1)) # "Isola" a entrada da vez
            u_t = w.T@x_t
            y_t = sinal(u_t[0,0])
            d_t = y_treino[t,0]
            diff = int(d_t-y_t)
            w = w + diff*x_t*LR
            if(d_t!=y_t):
                erro = True
                x2 = -x1*(w[1,0] / w[2,0]) + w[0,0] / w[2,0]
    #plotar()
    vp = 0
    vn = 0
    fn = 0
    fp = 0
    for i in range(x_teste.shape[1]):
        x_t = x_teste[:, i].reshape((p+1,1))
        u_t = w.T@x_t
        y_t = sinal(u_t[0,0])
        y_real = y_teste[i, 0]
        if (y_t == y_real):
            if (y_t > 0):
                vp = vp + 1
            else:
                vn = vn + 1
        else:
            if y_t < 0 and y_real > 0:
                fn = fn + 1;
            if y_t > 0 and y_real < 0:
                fp = fp + 1
    acuracia = ((vp + vn) / x_teste.shape[1]) * 100
    sensibilidade = (vp/(vp + fn)) * 100
    especificidade = (vn/(vn + fp)) * 100
    #print(f"Acurácia da rodada {r+1}: {acuracia:.2f}%")
    #print(f"Sensibilidade da rodada {r+1}: {sensibilidade:.2f}%")
    #print(f"Especificidade da rodada {r+1}: {especificidade:.2f}%\n")

