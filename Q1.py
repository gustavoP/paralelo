
# coding: utf-8

# # 1 - Regressão Linear com uma Variável
# 
# O primeiro passo é carregar os dados, a função "importarDados" foi criada como um facilitador na hora de importar os dados. Como entrada temos o endereço do arquivo e o nome que queremos dar as colunas dos dados, e como saída temos as características $X$ já com a primeira coluna de [1] e o vetor de rótulo $\vec{y}$

# In[11]:


import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D #para o 3d da função custo

import pymp
import random

# In[12]:


def importarDados(filepath,names):
    path = os.getcwd() + filepath  
    data = pd.read_csv(path, header=None, names=names)
    # adiciona uma coluna de 1s referente a variavel x0
    data.insert(0, 'Ones', 1)
    
    # separa os conjuntos de dados x (caracteristicas) e y (alvo)
    cols = data.shape[1]  
    X = data.iloc[:,0:cols-1]  
    y = data.iloc[:,cols-1:cols]
    
    # converte os valores em numpy arrays
    X = np.array(X.values)  
    y = np.array(y.values)
    
    return X,y


# In[13]:


filepath = "/ex1data1.txt"
X,y = importarDados(filepath,["Population","Profit"])


# # Visualizando os dados de entrada
# O gráfco mostra como estão dispersos os dados de entrada entre a única característica. Os dados que estamos visualizando são referente a lucro de um rede de food truck em diversas cidades separadas por população. A característica do problea é a população da cidade em base de $10^4$ habitantes, no eixo das abscissa e os dados de saída no eixo das ordenadas que representa o lucro com base de $\$ 10^4 $.
# 

# In[4]:


plt.scatter(X[:,1], y, color='red', marker='x')
plt.title('População da cidade x Lucro da filial')
plt.xlabel(r'População da cidade ($10^4$)')
plt.ylabel(r'Lucro ($\$10^4$)')
plt.show()


# # Gradiente Descendente
# Temos como objetivo principal fazer predições sobre o problema, dado uma cidade com uma determinada população queremos saber quanto de lucro será arrecadado com o food truk.
# 
# Utilizamos o Gradiente descendente para minimizar uma função custo $J(\theta)$ que melhor ajustará uma reta, nesse caso, aos dados fornecidos.
# A formulação será escrita em sua forma vetorial que foi mais próximo do que foi implementado no código.
# 
# $$ h_\theta(x) = \vec{h} = \theta^TX$$
# 
# $$ J(\theta) = \frac{1}{2m}(\vec{h}-\vec{y})^T*(\vec{h}-\vec{y})$$
# 
# Aplicando o gradiente descendente atualizamos o valor de $\theta$ para:
# 
# $$\vec{\theta} = \vec{\theta} -\alpha \nabla J(\theta)  =  \vec{\theta} - \frac{\alpha}{m}X^T*(\vec{h}-\vec{y}) $$
# 
# Sendo $m$ ao tamanho do conjunto de treinamento, $X$ a matriz de características sendo a primeiro coluna de $[1]$, $\alpha$ a taxa de aprendizado

# In[5]:


def custo_reglin_uni(X, y, theta):

    # Quantidade de exemplos de treinamento
    m = len(y)

    # Computar a função do custo J
    J = (np.sum((X.dot(theta) - y)**2)) / (2 * m)

    return J


# In[6]:


def gd_reglin_uni(X, y, alpha, epochs, theta = np.array([0,0], ndmin = 2).T):

    m = len(y)

    cost = np.zeros(epochs)

    for i in range(epochs):

        h = X.dot(theta)

        loss = h - y

        gradient = X.T.dot(loss) / m

        theta = theta - (alpha * gradient)

        cost[i] = custo_reglin_uni(X, y, theta = theta)

    return cost[-1], theta

def gd_reglin_uni_parallel(X, y, alpha, epochs, theta = np.array([0,0], ndmin = 2).T, proc = 4):
    n_times = proc
    costP = pymp.shared.array([n_times,1])
    thetaP = pymp.shared.array([n_times,2])

    m = len(y)

    with pymp.Parallel(4) as p:
        for j in p.range(0, n_times):
            cost = np.zeros(epochs)

            for i in range(epochs):

                h = X.dot(theta)

                loss = h - y

                gradient = X.T.dot(loss) / m

                theta = theta - (alpha * gradient)

                cost[i] = custo_reglin_uni(X, y, theta = theta)

                if(i==epochs-1):
                    costP[j] = cost[-1]
                    thetaP[[j]] = theta.T


    return costP.mean(0)[0], thetaP.mean(0).reshape(2,1)



# Foi fornecido o valor do custo para $\theta_0 = [0,0]^T$ é de 32.07, constatamos isso no código abaixo.

# In[7]:


print("Custo = {:.02f}, para theta = [0,0]'".format(custo_reglin_uni(X,y,np.array([[0],[0]]))))


# # Estimando Lucro
# Com nosso modelo treinado podemos predizer qual será o lucro para uma população de 35 mil e 70 mil habitantes.

# In[8]:


custo, theta = gd_reglin_uni(X,y,0.01,5000,np.array([[0],[0]]))
print("O lucro estimado de uma população de {} é ${:.0f}, e para uma população de {} o lucro é de ${:.0f}" 
    .format(35000,(theta[0,0]+theta[1,0]*3.5)*10000,70000,(theta[0,0]+theta[1,0]*7)*10000))

custo, theta = gd_reglin_uni_parallel(X,y,0.01,5000,np.array([[0],[0]]))
print("Paralelo ->O lucro estimado de uma população de {} é ${:.0f}, e para uma população de {} o lucro é de ${:.0f}" 
    .format(35000,(theta[0,0]+theta[1,0]*3.5)*10000,70000,(theta[0,0]+theta[1,0]*7)*10000))

# # Visualizando modelo
# Com os valores de theta calulado pelo gradiente descendente podemos traçar a reta $X*\theta$ para observar como nosso modelo faz a predição
# 

# In[9]:


plt.scatter(X[:,1], y, color='red', marker='x', label='Training Data')

t = np.arange(0, 25, 1)
line = theta[0] + (theta[1]*t)

plt.plot(t, line, color='blue', label='Linear Regression')
plt.axis([4, 25, -5, 25])
plt.title('População da cidade x Lucro da filial')
plt.xlabel('População da cidade (10k)')
plt.ylabel('Lucro (10k)')
plt.legend()
plt.show()


# # Plotando a função custo
# Podemos visualizar a função custo já que temos apenas 2 parâmetros: $\theta_0$ e $\theta_1$, para isso é feito um meshgrid e calculado para cada item o valor de $J(\theta)$ e utilizado para que seja plotado a superficie. Essa curva prova que somente existe um mínimo para a função custo

# In[10]:


theta0 = np.arange(-10, 10, 0.01)
theta1 = np.arange(-1, 4, 0.01)

# Comandos necessários para o matplotlib plotar em 3D
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plotando o gráfico de superfície
#theta0, theta1 = np.meshgrid(theta0, theta1)
#J=np.zeros(theta0.shape)
#for i in range(theta0.shape[0]):
#    for j in range(theta0.shape[1]):
#        t = np.array([[theta0[i][j]],[theta1[i][j]]])
#        J[i][j] = custo_reglin_uni(X,y,t)

#surf = ax.plot_surface(theta0, theta1, J)
#plt.xlabel(r'$\theta_0$')
#plt.ylabel(r'$\theta_1$')
#plt.title(r'Função custo $J(\theta)$')
#plt.show()

