import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def embaralhar_dados():
    seed = np.random.permutation(n)
    x_random = x[:, seed]
    y_random = y[seed, :]

    x_treino = x_random[:, 0 : int(n * 0.8)]
    y_treino = y_random[0 : int(n * 0.8), :]

    x_teste = x_random[:, int(n * 0.8):]
    y_teste = y_random[int(n * 0.8):, :]

    return (x_treino, y_treino, x_teste, y_teste)

def EQM(x_treino, y_treino, w):
    eqm = 0
    for t in range(x_treino.shape[1]):
        x_t = x_treino[:,t].reshape(p+1,1)
        u_t = w.T@x_t
        d_t = y_treino[t,0]
        eqm = eqm + (d_t - u_t)**2
    return eqm/(2*x_treino.shape[1])

pior_acuracia = 100
pior_mc = np.zeros((2,2), dtype=int)
melhor_mc = np.zeros((2,2), dtype=int)
melhor_acuracia = 0
medias = np.empty((3, 0))
rodadas = 100
rodadas = 100
for r in range(rodadas):
    (x_treino, y_treino, x_teste, y_teste) = embaralhar_dados()

    #w = np.zeros((p+1, 1))
    w = np.random.rand(p+1, 1)
    x1 = np.linspace(-12, 12, n)
    x2 = np.zeros((n,))
    EQM1 = 1
    EQM2 = 0
    LR = 1e-3
    pr = 1e-5

    epoch = 0
    maxEpoch = 5000
    while(epoch < maxEpoch and abs(EQM1-EQM2)>pr):
        EQM1 = EQM(x_treino,y_treino,w)
        for t in range(x_treino.shape[1]):
            x_t = x_treino[:,t].reshape(p+1,1)
            u_t = w.T@x_t
            d_t = y_treino[t,0]
            e_t = (d_t-u_t)
            w = w + e_t*x_t*LR
        epoch+=1
        x2 = -x1 * (w[1, 0] / w[2, 0]) + w[0, 0] / w[2, 0]
        EQM2 = EQM(x_treino,y_treino,w)
    #plotar()
    mc = np.zeros((2,2), dtype=int)
    for i in range(x_teste.shape[1]):
        x_t = x_teste[:, i].reshape(p+1,1)
        u_t = w.T@x_t
        y_t = sinal(u_t[0, 0])
        y_real = y_teste[i, 0]

        if (y_t == y_real):
            if (y_t > 0):
                mc[0,0] = mc[0,0] + 1 # Verdadeiro Positivo (VP)
            else:
                mc[1,1] = mc[1,1] + 1 # Verdadeiro Negativo (VN)
        else:
            if y_t > 0 and y_real < 0:
                mc[0,1] = mc[0,1] + 1 # Falso Positivo (FP)
            if y_t < 0 and y_real > 0:
                mc[1,0] = mc[1,0] + 1 # Falso Negativo (FN)

    acuracia = ((mc[0,0] + mc[1,1]) / x_teste.shape[1]) * 100
    sensibilidade = (mc[0,0]/(mc[0,0] + mc[1,0])) * 100
    especificidade = (mc[1,1]/(mc[1,1] + mc[0,1])) * 100
    medias = np.concatenate((medias, np.array([acuracia, sensibilidade, especificidade]).reshape(3, 1)), axis=1)
    #print(f"Acurácia da rodada {r+1}: {acuracia:.2f}%")
    #print(f"Sensibilidade da rodada {r+1}: {sensibilidade:.2f}%")
    #print(f"Especificidade da rodada {r+1}: {especificidade:.2f}%\n")
    if (acuracia > melhor_acuracia):
        melhor_acuracia = acuracia
        print("Melhor acurácia", melhor_acuracia)
        melhor_mc = mc
    if (acuracia < pior_acuracia):
        pior_acuracia = acuracia
        print("Pior acurácia", pior_acuracia)
        pior_mc = mc

print("\n\n=-=-=-=-=-Acurácia-=-=-=-=-=")
print("media", np.mean(medias[0, :]))
print("máximo", np.max(medias[0, :]))
print("minimo", np.min(medias[0, :]))
print("desvio padrão", np.std(medias[0, :]))

print("\n\n=-=-=-=-=-Sensibilidade-=-=-=-=-=")
print("media", np.mean(medias[1, :]))
print("máximo", np.max(medias[1, :]))
print("minimo", np.min(medias[1, :]))
print("desvio padrão", np.std(medias[1, :]))

print("\n\n=-=-=-=-=-Especificidade-=-=-=-=-=")
print("media", np.mean(medias[2, :]))
print("máximo", np.max(medias[2, :]))
print("minimo", np.min(medias[2, :]))
print("desvio padrão", np.std(medias[2, :]))

sns.heatmap(pior_mc, annot=True, fmt="d", cbar=False)
plt.show()
sns.heatmap(melhor_mc, annot=True, fmt="d", cbar=False)
plt.show()