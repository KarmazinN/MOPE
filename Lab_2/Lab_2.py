import numpy as np
m = 6

Y_max = (30-10)*10  # Y_max = 200
Y_min = (20-10)*10  # Y_max = 100
xn = [[-1, -1], [-1, 1], [1, -1]]
x1_min, x1_max, x2_min, x2_max = -25, -5, -30, 45

D_odn = True

while D_odn:
    Y = np.array([[np.random.randint(Y_min, Y_max) for j in range(m)] for i in range(3)])

    Y_average = np.array([])
    for i in Y:
        Y_average = np.append(Y_average, sum(i)/m)

    Sigma = np.sqrt((2 * (2 * m - 2)) / (m * (m - 4)))

    D = np.array([])
    for i in Y:
        D = np.append(D, np.var(i))

    def F(a, b):
        if a >= b:return a / b
        else:return b / a

    Fuv = np.array([])
    Fuv = np.append(Fuv, F(D[0], D[1]))
    Fuv = np.append(Fuv, F(D[2], D[0]))
    Fuv = np.append(Fuv, F(D[2], D[1]))

    teta = []
    Ruv = []
    for i in Fuv:
        t = np.append(teta, ((m - 2) / m) * i)
        Ruv = np.append(Ruv, (abs(((m - 2) / m) * i - 1) / Sigma))

    kr = 2
    for i in Ruv:
        if i < kr:
            D_odn = False
        else:
            m += 1

print("M: ", m)
print("Матриця планування:\n", Y)
print("\nСредні значення:\n", np.round(Y_average, 3))
print("\nДисперсія в кожному рядку:\n", np.round(D, 3))

print("\nВідхилення:", np.round(Sigma, 3))

mx1 = (xn[0][0] + xn[1][0] + xn[2][0]) / 3
mx2 = (xn[0][1] + xn[1][1] + xn[2][1]) / 3
my = (Y_average[0] + Y_average[1] + Y_average[2]) / 3

a1 = (xn[0][0] ** 2 + xn[1][0] ** 2 + xn[2][0] ** 2) / 3
a2 = (xn[0][0] * xn[0][1] + xn[1][0] * xn[1][1] + xn[2][0] * xn[2][1]) / 3
a3 = (xn[0][1] ** 2 + xn[1][1] ** 2 + xn[2][1] ** 2) / 3
a11 = (xn[0][0] * Y_average[0] + xn[1][0] * Y_average[1] + xn[2][0] * Y_average[2]) / 3
a22 = (xn[0][1] * Y_average[0] + xn[1][1] * Y_average[1] + xn[2][1] * Y_average[2]) / 3

m02 = [[1, mx1, mx2,],[mx1, a1, a2],[mx2, a2, a3]]

b0 = (np.linalg.det([[my, mx1, mx2],[a11, a1, a2],[a22, a2, a3]])/np.linalg.det(m02))
b1 = (np.linalg.det([[1, my, mx2],[mx1, a11, a2],[mx2, a22, a3]])/np.linalg.det(m02))
b2 = (np.linalg.det([[1, mx1, my],[mx1, a1, a11],[mx2, a2, a22]])/np.linalg.det(m02))

Tx1 = abs(x1_max - x1_min) / 2
Tx2 = abs(x2_max - x2_min) / 2
x10 = (x1_max + x1_min) / 2
x20 = (x2_max + x2_min) / 2

a0 = b0 - (b1 * x10 / Tx1) - (b2 * x20 / Tx2)
a1 = b1 / Tx1
a2 = b2 / Tx2

yn1 = a0 + a1 * x1_min + a2 * x2_min
yn2 = a0 + a1 * x1_min + a2 * x2_max
yn3 = a0 + a1 * x1_max + a2 * x2_min

print("\nПеревірка:\n", np.round(np.array([b0-b1-b2, b0-b1+b2, b0+b1-b2]), 3))
print("\nНормоване рівняння регресії:\n" + f" Y = {round(b0, 3)} + {round(b1, 3)} * x1 + {round(b2, 3)} * x2")
print("\nНатуралізоване рівняння регресії:\n" + f" Y = {round(a0, 3)} + {round(a1, 3)} * x1 + {round(a2, 3)} * x2")
print("\nПеревірка:\n", np.round(np.array([yn1, yn2, yn3]), 3))
