# Projeto de Classificação de Dígitos MNIST

Este projeto utiliza modelos de Deep Learning para classificar imagens de dígitos manuscritos do dataset MNIST. A aplicação foi desenvolvida usando Flask para o backend e HTML/CSS/JavaScript para o frontend.

## Video de demonstração

[Video](https://drive.google.com/file/d/174s_E7TVl7LJ4Husfn1ywAB3kQoYRhYD/view?usp=sharing)

## Requisitos

Antes de iniciar, você precisará ter o Python instalado em sua máquina. Além disso, as seguintes bibliotecas são necessárias:

```markdown
- Flask
- Flask-CORS
- OpenCV
- NumPy
- TensorFlow/Keras
```

Você pode instalar todas as dependências necessárias com o seguinte comando:

```bash
pip install Flask Flask-CORS opencv-python numpy tensorflow
```

## Estrutura do Projeto

O projeto consiste nos seguintes arquivos principais:

- `app.py`: Arquivo principal que contém o servidor Flask e os endpoints da API.
- `index.html`: Página da web que permite ao usuário carregar imagens e visualizar as previsões.
- `modelo_mnist.h5` e `linear_model_mnist.h5`: Modelos pré-treinados que são carregados pelo servidor.

## Como Executar

1. **Clonar o Repositório**

   Primeiro, clone o repositório para sua máquina local usando o seguinte comando:

   ```bash
   git clone https://github.com/LuanRM1/convolutionXLinear.git
   ```

2. **Iniciar o Servidor**

   Navegue até o diretório do projeto e execute o arquivo `app.py` para iniciar o servidor:

   ```bash
   python3 app.py
   ```

   Isso iniciará o servidor Flask na porta 8000.

3. **Acessar a Interface Web**

   Abra um navegador e acesse `http://localhost:8000` para interagir com a aplicação web.

## Funcionalidades

- **Upload de Imagem**: Os usuários podem fazer upload de imagens de dígitos para classificação.
- **Seleção de Modelo**: É possível escolher entre o modelo padrão e o modelo linear para a inferência.
- **Exibição de Resultados**: Os resultados da classificação e o tempo de inferência são exibidos na interface web.

## Resultados Observados

Os resultados a seguir destacam o desempenho dos modelos Convolucional e Linear após o treinamento com o dataset MNIST.

### Modelo Convolucional (LeNet-5)

- **Tempo de Treinamento**: 39.22 segundos
- **Acurácia**: 98.96%

Este modelo utiliza uma arquitetura convolucional para reconhecer padrões espaciais mais complexos nos dados de imagem, resultando em uma alta acurácia.

### Modelo Linear

- **Tempo de Treinamento**: 9.27 segundos
- **Acurácia**: 98.01%

O modelo linear, embora mais simples e com tempo de treinamento significativamente menor, também apresenta uma excelente acurácia, tornando-o uma alternativa viável quando a velocidade de treinamento é uma prioridade.

### Análise Comparativa

A diferença de acurácia entre os modelos é menor que 1%, no entanto, o tempo de treinamento do modelo convolucional é mais de quatro vezes maior que o do modelo linear. Isso sugere que, para aplicações que requerem rapidez na fase de treinamento, o modelo linear pode ser uma escolha eficiente, especialmente se a perda marginal de acurácia for aceitável.
