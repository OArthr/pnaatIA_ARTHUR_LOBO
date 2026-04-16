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

Liste as principais bibliotecas utilizadas no projeto, preferencialmente
com suas versões.



## 3️⃣ Técnica de Otimização do Modelo

Explique qual técnica foi utilizada para otimizar o modelo no arquivo
`optimize_model.py`.



## 4️⃣ Resultados Obtidos

Informe o principal resultado obtido após o treinamento do modelo.



## 5️⃣ Comentários Adicionais (Opcional)

Utilize este espaço para comentar:
- Dificuldades encontradas  
- Decisões técnicas importantes  
- Limitações do modelo  
- Aprendizados durante o desafio


## 🆘 Suporte

Em caso de dúvidas:

- Consulte o material dos cursos EAD
- Leia atentamente este README
- Analise os logs das GitHub Actions
- Utilize os canais oficiais para contato com os instrutores

Boa sorte no processo seletivo.
****
