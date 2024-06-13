import time
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np

(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()

y_treino_cat = to_categorical(y_treino)
y_teste_cat = to_categorical(y_teste)

x_treino = x_treino / x_treino.max()
x_teste = x_teste / x_teste.max()

x_treino = x_treino.reshape(-1, 28, 28, 1)
x_teste = x_teste.reshape(-1, 28, 28, 1)

model = Sequential(
    [
        Conv2D(32, (5, 5), padding="same", activation="relu", input_shape=(28, 28, 1)),
        MaxPool2D(strides=2),
        Conv2D(48, (5, 5), padding="valid", activation="relu"),
        MaxPool2D(strides=2),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(84, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=Adam())

start_time = time.time()
historico = model.fit(
    x_treino, y_treino_cat, epochs=10, validation_split=0.1, batch_size=32
)
end_time = time.time()

training_time = end_time - start_time
print(f"Tempo para treinar modelo LeNet-5: {training_time} segundos")

model.save("modelo_mnist.h5")

test_loss, test_acc = model.evaluate(x_teste, y_teste_cat)
print(f"Acurácia do modelo: {test_acc}")

modelo_2 = load_model("modelo_mnist.h5")

predicao = modelo_2.predict(x_teste[6].reshape(1, 28, 28, 1))
print(predicao)
print(np.argmax(predicao))
