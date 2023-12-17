# PixelRNN-PixelCNN

Proyecto final del curso MDS7203 - Modelos Generativos Profundos. 

Este proyecto contiene la implementación de los modelos autorregresivos PixelRNN (RowLSTM, DiagonalLSTM) y PixelCNN. Además se realiza una comparación entre los tiempos de inferencia entre los distintos modelos. El proyecto se base en el siguiente paper: [Pixel Recurrent Neural Networks](https://arxiv.org/pdf/1601.06759v3.pdf).

Integrantes:
- Diego Dominguez
- Daniel Minaya

Para ejecutar el Demo.ipynb basta clonar el repositorio con todas las carpetas necesarias.

- **data**

  Contiene los archivos py que permiten cargar los datasets MNIST y CIFAR10.

- **models**

  Contiene los modelos entrenados que se utilizaron en el informe.

- **PixelCNN**

  Contiene la arquitectura de Pixel CNN.

- **PixelRNN**

  Contiene la arquitectura de Pixel RNN junto a los módulos RowLSTM y DiagLSTM.

- **utils**

  Contiene las funciones auxiliares para entrenar los modelos. En el módulo config.py se pueden editar algunos hiperparámetros. En el módulo maskedConv se encuentran las implementaciones de las máscaras convolucionales. En el módulo train se encuentran las funciones para entrenar y samplear.