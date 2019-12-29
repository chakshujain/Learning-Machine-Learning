import numpy as np
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt

x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
y_data = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 
plt.scatter(x_data,y_data)
plt.xlabel('x')
plt.ylabel('y')


# Using Sklearn 
def usingSklearn(x_data,y_data):

    plt.scatter(x_data,y_data)
    plt.xlabel('x')
    plt.ylabel('y')
        
    regresser = linear_model.LinearRegression()

    regresser.fit(x_data.reshape(-1,1),y_data.reshape(-1,1))
    x0 = (regresser.coef_)
    x1 = (regresser.intercept_)
    print("x0: ",x0)
    print("x1: ",x1)
    costfun = 0
    for i in range(10):
        costfun = costfun + (x0 + x1*x_data[i] - y_data[i])**2
    costfun = costfun/(2*10)
    print("Cost Function: ",costfun)
    x0 = x0[0][0]
    x1 = x1[0]
    x = np.linspace(0,10,2)
    y = x0 + x1*x
    plt.plot(x,y)
    plt.show()



# Using Algebra

x0 = 1
x1 = 1
for i in range(500):
    for j in range(10):
        k1 = x1*x_data[j] + x0 - y_data[j]
        k2 = (x1*x_data[j] + x0 - y_data[j])*x_data[j]

    temp =  x0 - ((0.001/10))*k1
    temp1 = x1 - ((0.001/10))*k2

    x0 = temp
    x1 = temp1

print("x0: ",x0)
print("x1: ",x1)

costfun = 0
for i in range(10):
    costfun = costfun + (x0 + x1*x_data[i] - y_data[i])**2
costfun = costfun/(2*10)
print("Cost Function: ",costfun)

x = np.linspace(0,10,2)
y = x0 + x1*x
plt.plot(x,y)
plt.show()


#Use Sklearn
usingSklearn(x_data,y_data)


