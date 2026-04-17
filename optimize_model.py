import tensorflow as tf

#insira seu código aqui

#==========================================================================================================
# Otimizando o modelo

# Carregando o modelo treinado
model = tf.keras.models.load_model('model.h5')

# Convertendo e aplicando o Dynamic Range Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Salvando o modelo otimizado
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Convertendo e aplicando quantização float16
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Salvando o modelo otimizado
with open('model_float16.tflite', 'wb') as f:
    f.write(tflite_model)
