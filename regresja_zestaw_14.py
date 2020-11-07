import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
a = np.loadtxt('dane14.txt')

x = a[:,[0]]
y = a[:,[1]]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)


# ax2+bx+c
c = np.hstack([X_train*X_train, X_train, np.ones(X_train.shape)])
v = np.linalg.pinv(c) @ y_train
e = sum((y_train - (v[0]*X_train*X_train + v[1]*X_train + v[2]))**2)/len(X_train)
print("blad szkoleniowy dla modelu ax2 + bx + c: ", e)
e_test = sum((y_test - (v[0]*X_test*X_test + v[1]*X_test + v[2]))**2)/len(X_test)
print("blad testowy dla modelu ax2 + bx + c: ", e_test)

# ax+b
c2 = np.hstack([X_train, np.ones(X_train.shape)])
v2 = np.linalg.pinv(c2) @ y_train
e2 = sum((y_train -(v2[0]*X_train + v2[1]))**2)/len(X_train)
print("blad szkoleniowy dla modelu ax + b: ", e2)
e2_test = sum((y_test -(v2[0]*X_test + v2[1]))**2)/len(X_test)
print("blad testowy dla modelu ax + b: ", e2_test)

# ax3+bx2+cx+d
c3 = np.hstack([X_train**3, X_train*X_train, X_train, np.ones(X_train.shape)])
v3 = np.linalg.pinv(c3) @ y_train
e3 = sum((y_train - (v3[0]*X_train**3 + v3[1]*X_train*X_train + v3[2]*X_train + v3[3]))**2)/len(X_train)
print("blad szkoleniowy dla modelu ax3+bx2+cx+d: ", e3)
e3_test = sum((y_test - (v3[0]*X_test**3 + v3[1]*X_test*X_test + v3[2]*X_test + v3[3]))**2)/len(X_test)
print("blad testowy dla modelu ax3+bx2+cx+d: ", e3_test)

plt.plot(x, y, 'ro')
plt.plot(X_train, y_train, 'g*')
plt.plot(x,v[0]*x*x + v[1]*x + v[2])
plt.plot(x,v2[0]*x + v2[1])
plt.plot(x,v3[0]*x**3 + v3[1]*x*x + v3[2]*x + v3[3])
plt.show()

