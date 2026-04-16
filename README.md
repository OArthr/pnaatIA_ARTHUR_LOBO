# Processo Seletivo – Intensivo Maker | AI

👤 Identificação: **Arthur Lobo Feitosa de Oliveira**

## 1️⃣ Resumo da Arquitetura do Modelo

No arquivo `train_model.py`, a arquitetura da **CNN** implementada consiste em:

### Extração de detalhes

* Duas camadas de convolução, utilizando um filtro kernel 3x3 e a função de ativação ReLU.

Elas são usadas para extrair características da imagem de input, destacando onde na imagem o padrão definido pelo filtro foi detectado mais intensamente.

* Entre as camadas de convolução, há uma camada de Pooling Máximo 2x2.

Usada para reduzir o tamanho do mapa de características obtido após a primeira camada de convolução, simplificando futuros cálculos e tornando o modelo mais robusto ao reduzir o risco de overfitting.

* Após a segunda convolução, realizamos a operação de flatten.

Para transformar o mapa de características em um vetor unidimensional, preparando-o para a rede neural.

### Classificação

* Uma camada densa interna de 128 neurônios.

Onde ocorre a mágica do modelo, analisando as características obtidas e retornando valores para a próxima camada, com base em pesos que são ajustados durante o treinamento, efetivamente aprendendo a classificar corretamente.

* Uma camada dropout de 128 neurônios e com regularização de 50%.

Onde alguns neurônios são desativados durante o treino para a rede aprender de forma mais distribuida, combatendo o overfitting.

* Uma última camada densa de apenas 10 neurônios.

Representando a camada de saída da rede, onde os valores finais em cada neurônio representam a escolha do classificador.


## 2️⃣ Bibliotecas Utilizadas

* Keras de Tensorflow: Principal biblioteca que permite utilizar os modelos e as camadas, bem como o otimizador e métodos para realizar o treinamento do modelo
* Pyplot de Matplotlib: Utilizado durante o desenvolvimento do programa para facilitar o entendimento e vizualisação dos resultados das épocas do treinamento.



## 3️⃣ Técnica de Otimização do Modelo

No arquivo `optimize_model.py`, a técnica utilizada para otimizar o modelo foi a Quantização Pós-Treinamento. Implementada no bloco:

```py
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```


## 4️⃣ Resultados Obtidos

Após as cinco épocas de treinamentro da CNN, obtivemos uma accuracy final entre 0.98 e 0.99, com valor de perda de apenas aproximadamente 0.04. Até no final da primeira época do treinamento, a precisão já estava acima de 90%, com aumentos menores a cada época.



## 5️⃣ Comentários Adicionais (Opcional)

### Dificuldades encontradas

Dificuldades em entender o fluxo de trabalho com as bibliotecas necessárias, sendo necessário buscar a documentação e seguir os exemplos definidos nos cursos EAD.

### Aprendizados durante o desafio

- Conhecer de forma mais aprofundada e prática o processo de criação e treinamento de um modelo de aprendizado de máquina. 

- Compreender todo o processo realizado para o treinamento e classificação, antes mesmo das camadas de neurônios, como a convolução e outras camadas e como são utilizadas para otimizar o modelo e combater o overtiffing. 

- Além disso, nas camadas de classificação também foi possível conhecer a camada de dropout, entendendo sua funcionalidade e sua importância para um aprendizado mais distribuido e robusto.