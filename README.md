# 🧠 Shallow Neural Network Implementation

This project provides a simple implementation of a **Shallow Neural Network (SNN)** from scratch using Python and NumPy.
It is designed to help you understand the core concepts of neural networks, including forward propagation, backpropagation, and training using gradient descent.

---

## 🚀 Features

* Simple Shallow Neural Network architecture

  * Input Layer → Hidden Layer → Output Layer
* Supports:

  * Binary classification tasks
  * Mean Squared Error for regression tasks (optional)
* Uses common activation functions:

  * ReLU
  * Sigmoid
  * Tanh
* Customizable hyperparameters:

  * Number of hidden neurons
  * Learning rate
  * Number of epochs

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/shallow-neural-network.git
cd shallow-neural-network
```

Install required packages:

```bash
pip install -r requirements.txt
```

*(Optional: No heavy dependencies required besides NumPy and Matplotlib for visualization.)*

---

## ⚡ Usage

1. Prepare your dataset (features `X` and labels `y`).

2. Initialize the neural network:

   ```python
   from shallow_nn import ShallowNeuralNetwork

   nn = ShallowNeuralNetwork(input_size=10, hidden_size=5, output_size=1, learning_rate=0.01)
   ```

3. Train the model:

   ```python
   nn.train(X_train, y_train, epochs=1000)
   ```

4. Make predictions:

   ```python
   predictions = nn.predict(X_test)
   ```

5. Evaluate performance using accuracy or other metrics.

---

## 📊 Example

```python
from shallow_nn import ShallowNeuralNetwork
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate toy dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
nn = ShallowNeuralNetwork(input_size=10, hidden_size=5, output_size=1, learning_rate=0.01)
nn.train(X_train, y_train, epochs=1000)

# Predict and evaluate
predictions = nn.predict(X_test)
accuracy = (predictions == y_test).mean()
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

---

## ✅ Advantages

* Easy to understand and modify
* Perfect for educational purposes
* Lightweight and no need for heavy frameworks

---

## ⚠️ Limitations

* Not suitable for large-scale datasets
* Cannot capture complex non-linear relationships as well as Deep Neural Networks
* Manual implementation — for production, consider TensorFlow or PyTorch

