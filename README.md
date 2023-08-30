# Identify Iris flower

This project consist of a Machine Learning model created to identify Iris flowers using an Advanced Machine Learning Algorithm (Softmax Regression Algorithm) with Neural Network.

## The problem

### Identify Irises

Irises have influenced the design of the French fleur-de-lis, are commonly used in the Japanese art of flower arrangement
known as Ikebana, and underlie the floral scents of the “essence of violet” perfume. They’re also the subject of
this well-known machine learning project, in which you must create an ML model capable of sorting irises based on five
factors into one of three classes: Iris Setosa, Iris Versicolour, and Iris Virginica.

To create a model capable of classifying each iris instance into the appropriate class based on four attributes:
sepal length, sepal width, petal length, and petal width.

Advanced Machine Learning algorithm, using multi-class classification neural networks model to identify iris flower

### Choosing the Machine Learning Algorithm

Looking at the problem at hand, it is a classification problem, but in this case, it's not a binary classification of
yes or no, 0 or 1. Since we are to identifying three different flowers, it's a multi-class classification problem.
The best algorithm to use will be Logistic Regression, which is probably the single most widely used classification algorithm in the world.

Just to note, Linear regression is not a good algorithm for classification problems, the algorithm is about predicting a
number but not possible types of outcomes in this case.

Since the outcome we want from the model is to decide among 3 possible outcomes, in addition to the algorithm of choice,
there are some other algorithms like decision trees, XGBoost, that can also be used. But I am going to use an advanced
learning algorithm coupled with Logistic Regression.

We would be using Logistic Regression but not the type that determines two possible outcomes. The actual advanced
machine learning algorithm I will use is the Softmax Classification algorithm. The softmax regression algorithm is a
generalization of logistic regression, which is a binary classification algorithm, to the multiclass classification context.

Our model will be built with a neural network with softmax output with some optimization applied.

Basically, if we are looking for 3 possible outcomes, we can encode the outcome using 0, 1, 2, i.e.

```python

flowers = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}
```

We expect the model the predict flowers by producing the output as encoded flower types.
