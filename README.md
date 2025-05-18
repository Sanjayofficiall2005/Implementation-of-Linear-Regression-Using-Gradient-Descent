# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**: Load necessary libraries for data handling, metrics, and visualization.

2. **Load Data**: Read the dataset using `pd.read_csv()` and display basic information.

3. **Initialize Parameters**: Set initial values for slope (m), intercept (c), learning rate, and epochs.

4. **Gradient Descent**: Perform iterations to update `m` and `c` using gradient descent.

5. **Plot Error**: Visualize the error over iterations to monitor convergence of the model.

## Program and Outputs:
```python
Program to implement the linear regression using gradient descent.
Developed by   : SANJAY M
RegisterNumber : 212223230187
```

```python
import pandas as pd
import numpy as np
```
```python
df = pd.read_csv("50_Startups.csv")
```

```python
df.head()
```

![image](https://github.com/user-attachments/assets/f3f07f6c-f734-4fca-b6c4-6e520471e9b0)


```python
df.tail()
```

![image](https://github.com/user-attachments/assets/a28e10e2-43c0-495b-8502-4fca2d4b86ae)


```python
df.info()
```

![image](https://github.com/user-attachments/assets/7643f4f4-f04f-4c47-b29b-67cbeef5fbaf)


```python
X = (df.iloc[1:,:-2].values)
y = (df.iloc[1:,-1].values).reshape(-1,1)
```

```python
print(X)
```

![image](https://github.com/user-attachments/assets/7c8fd303-e9cc-4063-94ac-92156aa749eb)


```python
print(y)
```

![image](https://github.com/user-attachments/assets/6c77d7e6-964b-4b51-89ef-06d5e0b938c9)


```python
from sklearn.preprocessing import StandardScaler
def multivariate_linear_regression(X1,Y):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    num_iters = 1000
    error = []
    learning_rate = 0.001
    
    for _ in range(num_iters):
        # Calculatenpredictions
        predictions = (X).dot(theta).reshape(-1,1)
        
        #Calculate errors
        errors = (predictions - Y).reshape(-1,1)
        
        #Upadte theta using gradient descent
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
        
        # Record the error for each iteration
        error.append(np.mean(errors ** 2))

    return theta, error, num_iters
```

```python
scaler = StandardScaler()
```

```python
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(y)
```

```python
print(X_scaled)
```

![image](https://github.com/user-attachments/assets/313fe356-8ed2-4827-9dcb-37bcd5a1b7f3)


```python
print(Y_scaled)
```

![image](https://github.com/user-attachments/assets/c8b2e70a-d214-4151-a49c-abd50f27d10e)


```python
# Train the model using scaled data
theta, error, num_iters = multivariate_linear_regression(X_scaled,Y_scaled)
```

```python
# Print the results
print("Theta:", theta)
print("Errors:", error)
print("Number of iterations:", num_iters)
```

![image](https://github.com/user-attachments/assets/38008b31-d2e0-46a8-aa2c-2d9f8c7a7147)


```python
type(error)
print(len(error))
```
![image](https://github.com/user-attachments/assets/2e9bfe19-4538-404e-b3a0-ab2ec0f4d1df)


```PYTHON
import matplotlib.pyplot as plt
plt.plot(range(0,num_iters),error)
```

![image](https://github.com/user-attachments/assets/59268074-24a3-4062-a71a-d65e25a29c17)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
