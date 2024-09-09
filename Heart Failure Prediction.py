import numpy as np
import pandas as pd
import sklearn.metrics as met
import sklearn.neighbors as ne
import matplotlib.pyplot as plt
import sklearn.linear_model as li
import sklearn.preprocessing as pp
import sklearn.neural_network as nn
import sklearn.model_selection as ms

def PrintReport(Model, trX, teX, trY, teY):
    trAc = Model.score(trX, trY)
    teAc = Model.score(teX, teY)
    trPr = Model.predict(trX)
    tePr = Model.predict(teX)
    trCR = met.classification_report(trY, trPr)
    teCR = met.classification_report(teY, tePr)
    print(f'{trAc = }')
    print(f'{teAc = }')
    print('_'*50)
    print(f'Test CR:\n{teCR}')
    print('_'*50)

np.random.seed(0)
plt.style.use('ggplot')

DF = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')
DF.drop(['time'], axis=1, inplace=True)

D = DF.to_numpy()

X = D[:, :-1]
Y = D[:, -1].reshape((-1, 1))

trX0, teX0, trY, teY = ms.train_test_split(X, Y, train_size=0.7, random_state=5)

Scaler = pp.MinMaxScaler(feature_range=(-1, +1))
trX = Scaler.fit_transform(trX0)
teX = Scaler.transform(teX0)

Zero = (trY == 0).reshape(-1)
One = (trY == 0).reshape(-1)

n0 = trY[Zero].size
n1 = trY[One].size

W = {0: n1/(n0+n1), 1: n0/(n0+n1)}

LR = li.LogisticRegression(random_state=0, class_weight=W)
LR.fit(trX, trY)
PrintReport(LR, trX, teX, trY, teY)

# KNN = ne.KNeighborsClassifier(n_neighbors=5, weights='distance')
# KNN.fit(trX, trY)
# PrintReport(KNN, trX, teX, trY, teY)

trX2 = []
trY2 = []

for i in range(trX.shape[0]):
    x = trX[Zero][np.random.randint(n0)]
    trX2.append(x)
    x = trX[One][np.random.randint(n1)]
    trX2.append(x)
    trY2.append([0])
    trY2.append([1])

trX2 = np.array(trX2)
trY2 = np.array(trY2)

MLP = nn.MLPClassifier(hidden_layer_sizes=(50), activation='relu', max_iter=100, random_state=0)
MLP.fit(trX2, trY2)
PrintReport(MLP, trX, teX, trY, teY)