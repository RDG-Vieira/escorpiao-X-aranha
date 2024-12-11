import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Definir parâmetros
IMG_SIZE = 150  # Tamanho da imagem (ajuste conforme necessário)
BATCH_SIZE = 32
EPOCHS = 20

# Usando ImageDataGenerator para pré-processar as imagens
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normaliza os valores das imagens para o intervalo [0, 1]
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Diretórios de treino e teste (ajuste para o seu diretório)
train_directory = 'C:\\Users\\faiel\\OneDrive\\Área de Trabalho\\escorpiao_aranha\\train'
test_directory = 'C:\\Users\\faiel\\OneDrive\\Área de Trabalho\\escorpiao_aranha\\test'

train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'  # Como temos duas classes (escorpião vs aranha)
)

test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Criando o modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.1),  # Para evitar overfitting
    Dense(1, activation='sigmoid')  # Saída binária (escorpião ou aranha)
])

# Compilando o modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Definir EarlyStopping para evitar overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Treinando o modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=test_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping]
)

# Avaliar o desempenho do modelo
test_loss, test_acc = model.evaluate(test_generator)
print(f'Acurácia no teste: {test_acc:.2f}') 
# -> ATÉ O MOMENTO A ACURÁCIA FOI DE APENAS 0.72, QUERIA AUMENTAR PARA 0.85

# Visualizando as curvas de treinamento e validação
plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

# Função para classificar uma nova imagem
def classificar_imagem(imagem_path):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(imagem_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)
    
    prediction = model.predict(img_array)
    if prediction < 0.5:
        print("A imagem é de um aranha.")
    else:
        print("A imagem é de uma escorpião.")

'''Testando com uma nova imagem ELE AINDA NÃO CONSEGUE RECONHECER UMA ARANHA SE A IMAGEM FOR MUITO GRANULADA, ESSE É O PROXIMO PASSO PARA RESOLVER NO PROGRAMA.'''
classificar_imagem('C:\\Users\\faiel\\OneDrive\Área de Trabalho\\escorpiao_aranha\\validacao\\escorpiao\\escorpiao.jpg')


