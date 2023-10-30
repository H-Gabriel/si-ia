import numpy as np 
import matplotlib.pyplot as plt 
 
# função de ativação sinal. poderia ser 0 ou 1. classificação binária 
def sinal(x): 
    if x>=0: 
        return  1 
    else: 
        return -1 
 
# pontos aleatórios. só toma cuidado se tu for mudar pq eles precisam ser separáveis, se não forem, o algoritmo não para. 
X = np.array([[1,1],[0,1],[0,2],[1,0],[2,2],[4,1.5],[1.5,6],[3,5],[3,3],[6,4]]) 
##X eh a minha matriz de pontos, todos os pontos estao aqui com [x,y] 
# as labels. 1 é um conjunto, -1 outro conjunto 
y = np.array([1,1,1,1,1,-1,-1,-1,-1,-1]) 
##Y sao as minhas respostas esperadas, logo eh um algoritmo de treinamento supervisionado 
# só atribuindo cor para os conjuntos quando tem :(sem nada aqui) é pq é até o fim 
plt.scatter(X[0:5, 0], X[0:5, 1], color = 'brown' ) 
plt.scatter(X[5: , 0], X[5: , 1], color = 'orange') 
 
 
X  = X.T 
print(X) 
 
# adicionando os 1's. se não adicionar, todos eles vão passar pela origem 
X  = np.concatenate(( -np.ones((1,10) ), X), axis=0) 
 
y.shape = (len(y), 1) 
#W  = np.random.random_sample((3,1))-.5 
W = np.array([[0],[0],[0]]) 
x1 = np.linspace(-2, 8, 10) 
#x2 = -x1*(W[1,0] / W[2,0]) + W[0,0] / W[2,0] 
x2 = np.zeros((10,)) 
 
# plotar a primeira linha. o perceptron evolui a partir dela. se tu quiser ver alguma mudanca, modifica o learning rate la embaixo LR 
plt.plot(x1, x2, color='red', linewidth=3) 
 
# limites do grafico 
plt.xlim(-1, 7) 
plt.ylim(-1, 7) 
 
# leraning rate(passo de aprendizagem) 
LR = 0.01 
 
# não tem condição de parada. vai rodar até não ter erro. 
Erro = True 
while( Erro ): 
    Erro = False 
    for t in range( X.shape[1] ): 
        x_t = X[ : , t].reshape(3,1) 
        print(W.T) 
        print(x_t) 
        u_t = W.T @ x_t #u = W * xK  
         
        y_t = sinal(u_t[0,0]) # calculando G(u) 
        d_t = y[t,0] #D sired value for this colum of test 
        W = W + LR*(d_t - y_t)*x_t 
        if( (d_t - y_t) != 0): #se houver erro: 
            Erro = True 
            x2 = -x1*(W[1,0] / W[2,0]) + W[0,0] / W[2,0] 
 
            # vai ficar plotando toda hora e ficar pausado por 0.05 
            plt.plot(x1, x2, color = 'k', alpha = 0.4) 
            plt.pause(0.2) 
             
x2 = -x1*(W[1,0] / W[2,0]) + W[0,0] / W[2,0] 
plt.plot(x1, x2, color = 'blue', linewidth=3) 
plt.pause(0.05) 
plt.show()