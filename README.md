# Neural Network from Scratch with NumPy

Implementación de una red neuronal completamente desde cero utilizando **NumPy**, sin emplear frameworks de deep learning como **TensorFlow** o **PyTorch**.

El objetivo de este proyecto es comprender los **principios fundamentales de las redes neuronales** implementando manualmente todos sus componentes.

Entre ellos:

- Forward propagation
- Backpropagation
- Descenso por gradiente
- Inicialización de pesos (He initialization)
- Función de pérdida Binary Cross Entropy
- Activaciones ReLU y Sigmoid

La red neuronal se entrena para resolver un problema clásico de **clasificación no lineal** usando el dataset de **espirales entrelazadas**.

---

# Visión general del proyecto

Hoy en día las redes neuronales suelen utilizarse a través de librerías que abstraen su funcionamiento interno.  
Este proyecto elimina esa abstracción y muestra **cómo funciona realmente una red neuronal por dentro**.

Se implementa manualmente:

- una red neuronal multicapa (MLP)
- propagación hacia delante
- retropropagación del error
- cálculo manual de gradientes
- actualización de parámetros mediante descenso por gradiente
- operaciones vectorizadas usando NumPy

---

# Arquitectura de la red

Arquitectura utilizada en el experimento:

```
Capa de entrada: 2 neuronas
Capa oculta 1: 64 neuronas (ReLU)
Capa oculta 2: 64 neuronas (ReLU)
Capa oculta 3: 64 neuronas (ReLU)
Capa de salida: 1 neurona (Sigmoid)
```

---

# Componentes matemáticos

## Forward propagation

Para cada capa se calcula:

```
z = XW + b
a = activation(z)
```

Las capas ocultas utilizan **ReLU**:

```
ReLU(x) = max(0, x)
```

La capa de salida utiliza **Sigmoid**:

```
σ(x) = 1 / (1 + e^{-x})
```

---

## Función de pérdida

Se utiliza **Binary Cross Entropy**, adecuada para clasificación binaria:

```
L = -1/N Σ [y log(p) + (1-y) log(1-p)]
```

---

## Backpropagation

Los gradientes se calculan manualmente utilizando la **regla de la cadena**.

Cuando se usa sigmoid en la salida con Binary Cross Entropy, el gradiente se simplifica a:

```
dL/dA = (y_pred - y_true) / N
```

Esto permite retropropagar el error capa por capa.

---

# Inicialización de pesos

Los pesos se inicializan usando **He initialization**, adecuada para activaciones ReLU:

```
W ~ N(0, √(2 / fan_in))
```

Los bias se inicializan en cero.

---

# Dataset

El modelo se entrena utilizando el **dataset de espirales**, un problema clásico de clasificación no lineal.

Este dataset consiste en dos espirales entrelazadas que representan dos clases diferentes.  
Es un buen ejemplo para probar la capacidad de una red neuronal de aprender **fronteras de decisión complejas**.

---

# Resultados

La red neuronal aprende una frontera de decisión no lineal capaz de separar correctamente ambas espirales.

El entrenamiento muestra:

- disminución progresiva de la función de pérdida
- mejora en la precisión de clasificación
- una frontera de decisión compleja aprendida por la red

---

# Tecnologías utilizadas

- Python
- NumPy
- Matplotlib

---

# Complejidad computacional

El entrenamiento de la red tiene una complejidad aproximada:

O(N · L · d²)

donde:

- N es el número de muestras
- L el número de capas
- d el número medio de neuronas por capa
---

# Objetivo educativo

Este proyecto está pensado como un ejercicio para comprender **cómo funcionan realmente las redes neuronales**, implementando cada componente manualmente.

Antes de utilizar frameworks avanzados, es fundamental entender:

- cómo se calcula el forward
- cómo se derivan los gradientes
- cómo se propaga el error
- cómo se actualizan los pesos

---

# Posibles mejoras

Algunas extensiones posibles del proyecto:

- implementar optimizadores como Adam o RMSProp
- añadir regularización L2
- implementar batch training
- añadir más funciones de activación
- separar el código en módulos (`forward`, `backward`, `train`, `predict`)
