from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import pandas as pd
import numpy as np

data = pd.read_excel('kelvin_celsius.xlsx')
entradas = np.array(data['Kelvin'],dtype='float32')
salidas = np.array(data['Celsius'],dtype='float32')

modelo = Sequential()

modelo.add(Dense(4,input_dim=1,activation='linear'))
modelo.add(Dense(4,activation='linear'))
modelo.add(Dense(1,activation='linear'))

adam = Adam()

modelo.compile(loss='mse',optimizer=adam,metrics='accuracy')

modelo.fit(entradas,salidas,epochs=3700)
print('-- Predicciones --')
print(modelo.predict(entradas))
print('------------------')

print('-- Valores reales --')
print(salidas)
print('--------------------')



