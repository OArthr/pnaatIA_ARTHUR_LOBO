
#insira seu código aqui

#==========================================================================================================
# Preparando o dataset
 
# Baixando o dataset mnist
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
# Dividindo o dataset de treino em treino e validação de forma balanceada
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.25)
 
# Checando quantidade de imagens do dataset
print('Quantidade de imagens de treino:', x_train.shape[0])
print('Quantidade de imagens de validação:', x_val.shape[0])
print('Quantidade de imagens de test:', x_test.shape[0])
 
 
# Plotando quantidade de imagens de cada dígito
from matplotlib import pyplot

""" fig, ax = pyplot.subplots()
rects1 = ax.bar(counterTrain.keys(), counterTrain.values(), label='Treino')
rects2 = ax.bar(counterVal.keys(), counterVal.values(), label='Validação')
rects3 = ax.bar(counterTest.keys(), counterTest.values(), label='Teste')
 
ax.set_title('Imagens por dígito')
ax.set_ylabel('Quantidade de imagens')
ax.set_xlabel('Dígito')
ax.legend()
pyplot.show()  """


#==========================================================================================================
# Formatando o dataset para funcionar como entrada do Keras 

# As imagens de entradas precisam estar em um array de 4 dimensões
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Cada imagem precisa ter dimensão x, y e z

input_shape = (28, 28, 1)

# Convertento valores dos pixels para float
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
 
# Normalizando os valores dos pixels
x_train /= 255
x_val /= 255
x_test /= 255


#==========================================================================================================
# Importando Keras e suas operações

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D

# criando a CNN
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),

    GlobalAveragePooling2D(),

    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# Resumo do modelo
model.summary()


#==========================================================================================================
# Definindo otimizador, função de perda e métrica de eficiência.

from keras.optimizers import Adam

adamOptimizer = Adam(learning_rate=0.001)

model.compile(optimizer=adamOptimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Efetuando o treinamento de 5 épocas com o dataset de treino e validando no dataset de validação
history = model.fit(x=x_train, y=y_train, validation_data=(x_val,y_val), epochs=5, batch_size=16, shuffle=False)

# Plotando o histórico de treino

# Histórico de acurácia
pyplot.plot(history.history['accuracy'])
pyplot.plot(history.history['val_accuracy'])
pyplot.title('Acurácia do modelo no treino e validação')
pyplot.ylabel('Acurácia')
pyplot.xlabel('Época')
pyplot.legend(['Treino', 'Validação'], loc='upper left')
pyplot.show()

# Histórico de perda
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('Perda do modelo no treino e validação')
pyplot.ylabel('Perda')
pyplot.xlabel('Época')
pyplot.legend(['Treino', 'Validação'], loc='upper left')
pyplot.show()


#==========================================================================================================
# Avaliando a CNN treinada
score = model.evaluate(x_test, y_test)

print('\nPerda:{:.3f}\nAcurácia:{}'.format(score[0], score[1]))

# Obtendo matriz de confusão
from sklearn.metrics import confusion_matrix
import numpy as np
y_pred = model.predict(x_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
y_true = y_test

cm = confusion_matrix(y_true, y_pred_classes)

# Plotando a matriz de confusão como heatmap
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
pyplot.title('Matriz de Confusão')
pyplot.xlabel('Predito')
pyplot.ylabel('Real')
pyplot.show()


#==========================================================================================================
# Salvando o modelo treinado
model.save('model.h5')

model.summary() 