import cv2
import numpy as np

folderRoot = 'faces\\'
individual = ['an2i','at33','boland', 'bpm', 'ch4f', 'cheyer', 'choon', 'danieln', 'glickman', 'karyadi', 'kawamura', 'kk49', 'megak', 'mitchell', 'night', 'phoebe', 'saavik', 'steffi', 'sz24', 'tammo'] #os 20 sujeitos no conjunto de dados.
expressoes =['_left_angry_open', '_left_angry_sunglasses', '_left_happy_open','_left_happy_sunglasses', '_left_neutral_open', '_left_neutral_sunglasses', '_left_sad_open', '_left_sad_sunglasses', '_right_angry_open', '_right_angry_sunglasses', '_right_happy_open', '_right_happy_sunglasses', '_right_neutral_open', '_right_neutral_sunglasses', '_right_sad_open', '_right_sad_sunglasses', '_straight_angry_open', '_straight_angry_sunglasses', '_straight_happy_open', '_straight_happy_sunglasses', '_straight_neutral_open', '_straight_neutral_sunglasses', '_straight_sad_open', '_straight_sad_sunglasses' ,'_up_angry_open', '_up_angry_sunglasses', '_up_happy_open', '_up_happy_sunglasses', '_up_neutral_open', '_up_neutral_sunglasses', '_up_sad_open', '_up_sad_sunglasses'] 
QtdIndividuos = len(individual)
QtdExpressoes = len(expressoes) 
Red = 30 #Tamanho do redimensionamento da imagem.
X = np.empty((Red*Red,0))
Y = np.empty((QtdIndividuos,0))

for i in range(QtdIndividuos):
    for j  in range(QtdExpressoes):
        path = folderRoot+individual[i]+'\\'+individual[i]+expressoes[j]+'.pgm'
        PgmImg = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        ResizedImg =cv2.resize(PgmImg,(Red,Red))
        
        VectorNormalized = ResizedImg.flatten('F')
        ROT = -np.ones((QtdIndividuos,1))
        ROT[i,0] = 1
        
        #cv2.imshow("Foto",PgmImg)
        #cv2.waitKey(0)

        VectorNormalized.shape = (len(VectorNormalized),1)
        X = np.append(X,VectorNormalized,axis=1)
        Y = np.append(Y,ROT,axis=1)
p, n = X.shape
c = Y.shape[0]   
print("Shape de x:", X.shape)
print("Shape de y:", Y.shape)

X = 2 * (X / 255) - 1 # Normalização
X = np.concatenate((-np.ones((1,n)), X))
L = 3 # qtd camadas ocultas
q = 30 # qtd neuronios camadas ocultas
m = 20 # qtd neuronios camada de saida
i = [None] * (L + 1) # Vetores de entrada de cada L-ésima camada
y = [None] * (L + 1) # Vetores de saída após aplicação da função de ativação em cada neurônio da L-ésima camada
delta = [None] * (L + 1) # ??? não sei o que é
LR = 1e-3
erro_max = 1e-2
max_epoch = 10000

# De acordo com o slide 200
W = []
for j in range(L+1):
    if j == 0:
        W.append(np.random.rand(q, p + 1) - 0.5) # Primeira camada
    elif j == L:
        W.append(np.random.rand(m, q + 1) - 0.5) # última camada
    else: 
        W.append(np.random.rand(q, q + 1) - 0.5) # Camadas intermediárias

def g(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))

def g_linha(x):
    return 0.5 * (1 - g(x) ** 2)

def forward(amostra):
    for j in range(L + 1):
        #print("Shape da matriz W da vez:", W[j].shape)""
        if j == 0:
            #print("Shape da amostra da vez:", amostra.shape)
            i[j] = W[j] @ amostra
            y[j] = g(i[j])
        else:
            y_bias = np.concatenate((-np.ones((1,1)), y[j-1]))
            #print("Shape da y_bias da vez:", y_bias.shape)
            i[j] = W[j] @ y_bias
            y[j] = g(i[j])
            
def backward(amostra, rotulo):
    #print("Shape do rótulo recebido:", rotulo.shape)
    j = L
    Wb = [None] * (len(W) + 1)
    while j >= 0:
        if j == L:
            delta[j] = g_linha(i[j]) * (rotulo - y[j])
            y_bias = np.concatenate((-np.ones((1,1)), y[j-1]))
            #print("shape do delta da vez:", delta[j].shape)
            #print("shape do y_bias da vez:", y_bias.shape)
            #print()
            W[j] = W[j] + (LR * (delta[j] @ y_bias.T))
        elif j == 0:
            Wb[j + 1] = np.delete(W[j + 1].T, 0, 0)
            delta[j] = g_linha(i[j]) * (Wb[j+1] @ delta[j+1])
            #print("shape do delta da vez:", delta[j].shape)
            #print("shape da amostra da vez:", amostra.shape)
            #print()
            W[j] = W[j] + (LR * (delta[j] @ amostra.T))
        else:
            Wb[j + 1] = np.delete(W[j + 1].T, 0, 0)
            delta[j] = g_linha(i[j]) * (Wb[j+1] @ delta[j+1])
            y_bias = np.concatenate((-np.ones((1, 1)), y[j-1]))
            #print("shape do delta da vez:", delta[j].shape)
            #print("shape do y_bias da vez:", y_bias.shape)
            #print()
            W[j] = W[j] + (LR * (delta[j] @ y_bias.T))
        j-=1


def eqm(x_treino, y_treino):
    eqm = 0
    for k in range(int(n*0.8)):
        amostra = x_treino[:, k].reshape(p+1, 1)
        forward(amostra)
        rotulo = y_treino[:, k].reshape(c, 1)
        eqi = 0
        for j in range(y[L].shape[0]):
            eqi = eqi + (rotulo[j] - y[L][j][0])**2
        eqm = eqm + eqi
    eqm = eqm/(2*x_treino.shape[1])
    return eqm

x_treino = X[:, : int(n * 0.8)]
y_treino = Y[:, : int(n * 0.8)]
x_teste = X[:, int(n * 0.8) :]
y_teste = Y[:, int(n * 0.8) :]

valor_eqm = 1
epoch = 0
while epoch < max_epoch and valor_eqm > erro_max:
    for k in range(int(n * 0.8)):
        amostra = x_treino[:, k].reshape(p+1, 1)
        forward(amostra)
        rotulo = y_treino[:, k].reshape(c, 1)
        backward(amostra, rotulo)
    valor_eqm = eqm(x_treino, y_treino)
    print(f"\rÉpoca: {epoch}, EQM: {valor_eqm}", end="")
    epoch+=1

acertos = 0
for k in range(int(n * 0.2)):
    amostra_teste = x_teste[:, k].reshape(p+1, 1)
    forward(amostra_teste)
    rotulo = y_treino[:, k].reshape(c, 1)
    if (np.argmax(y[L]) == np.argmax(rotulo)):
        acertos+=1
print("\nAcertos:", acertos)
print("Porcentagem:", acertos/int(n*0.2))