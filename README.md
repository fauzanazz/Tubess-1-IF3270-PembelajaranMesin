# 🧠 Feedforward Neural Network from Scratch

This repository contains a Feedforward Neural Network (FFNN) built **from scratch** using Python and NumPy. The project was developed as part of **Tugas Besar 1** for the **IF3270 Machine Learning** course at Institut Teknologi Bandung.

## 📌 Features

- ✅ Fully customizable neural network architecture
- ✅ Supports various activation functions:
  - Linear
  - ReLU
  - Sigmoid
  - Tanh
  - Softmax
- ✅ Multiple loss functions:
  - Mean Squared Error (MSE)
  - Binary Cross Entropy
  - Categorical Cross Entropy
- ✅ Weight initialization methods:
  - Zero Initialization
  - Random Uniform
  - Random Normal
- ✅ L1 and L2 Regularization
- ✅ Batch Gradient Descent
- ✅ Training/validation visualization (loss, weight/gradient distributions)
- ✅ Model saving and loading
- ✅ Evaluation and comparison with `sklearn`'s MLP

## 🧪 Experiments

The network is tested on the **MNIST** dataset using `fetch_openml`. Performance and training insights are documented in the report.

Visualizations include:
- Network structure with weight and gradient flow
- Loss curve per epoch
- Distribution plots of weights and gradients per layer


## ⚙️ Setup and Run the Program

1. **Clone repository:**
```bash
git clone https://github.com/username/nama-repo.git](https://github.com/fauzanazz/Tubess-1-IF3270-PembelajaranMesin.git 
```
2. **Running the program**
```bash
python main.py
```
or run the ```simple.ipynb``` 
3. **Installing all dependencies**
```bash
pip install -r requirements.txt
```

## 👨‍👩‍👧‍👦 Contributors
| Nama                   | NIM         | Contribution                                                                   |
|------------------------|-------------|--------------------------------------------------------------------------------|
| Auralea Alvinia S      | 13522148    | - Visualizing structure graph, weight distribution, gradient distribution      | 
|                        |             | - Input layer                                                                  |
| M. Fauzan Azhim        | 13522153    | - Regularization, normalization, activation function, loss function            |
|                        |             | - Back propagation, forward propagation                                        |
| Pradipta Rafa Mahesa   | 13522162    | - Creating safe and load                                                       |
|                        |             | - Weight init                                                                  |
