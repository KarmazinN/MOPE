import numpy as np
import random
import sklearn.linear_model as lm
from scipy.stats import f, t
from math import sqrt
from pyDOE2 import *

global counter
counter = 0

x_range = [(-4, 3), (-6, 10), (0, 3)]
xcp_min = round(sum([x_range[i][0] for i in range(len(x_range))]) / 3)
xcp_max = round(sum([x_range[i][1] for i in range(len(x_range))]) / 3)
y_min, y_max = 200 + xcp_min, 200 + xcp_max

def regression(x, b):
    return sum([x[i] * b[i] for i in range(len(x))])

def matrix(m, n):
    y = np.zeros(shape=(n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            y[i][j] = random.randint(y_min, y_max)

    no = 1
    x_norm = ccdesign(3, center=(0, no))
    x_norm = np.insert(x_norm, 0, 1, axis=1)

    for i in range(4, 11):
        x_norm = np.insert(x_norm, i, 0, axis=1)

    l = 1.215

    for i in range(len(x_norm)):
        for j in range(len(x_norm[i])):
            if x_norm[i][j] < -1:
                x_norm[i][j] = -l
            elif x_norm[i][j] > 1:
                x_norm[i][j] = l

    def inter_matrix(x):
        for i in range(len(x)):
            x[i][4] = x[i][1] * x[i][2]
            x[i][5] = x[i][1] * x[i][3]
            x[i][6] = x[i][2] * x[i][3]
            x[i][7] = x[i][1] * x[i][2] * x[i][3]
            x[i][8] = x[i][1] * x[i][1]
            x[i][9] = x[i][2] * x[i][2]
            x[i][10] = x[i][3] * x[i][3]

    inter_matrix(x_norm)

    x_natur = np.ones(shape=(n, len(x_norm[0])), dtype=np.float64)
    for i in range(8):
        for j in range(1, 4):
            if x_norm[i][j] == 1:
                x_natur[i][j] = x_range[j-1][1]
            else:
                x_natur[i][j] = x_range[j-1][0]
    x0 = [(x_range[i][1] + x_range[i][0]) / 2 for i in range(3)]
    dx = [x_range[i][1] - x0[i] for i in range(3)]

    for i in range(8, len(x_norm)):
        for j in range(1, 4):
            if x_norm[i][j] == 0:
                x_natur[i][j] = x0[j-1]
            elif x_norm[i][j] == l:
                x_natur[i][j] = l * dx[j-1] + x0[j-1]
            elif x_norm[i][j] == -l:
                x_natur[i][j] = -l * dx[j-1] + x0[j-1]

    inter_matrix(x_natur)
    y_aver = [sum(y[i]) / m for i in range(n)]

    print("Нормована матриця Х\n")
    for i in range(len(x_norm)):
        for j in range(len(x_norm[i])):
            print(round(x_norm[i][j], 3), end=' ')
        print()

    print("\nНатуралізована матриця Х\n")
    for i in range(len(x_natur)):
        for j in range(len(x_natur[i])):
            print(round(x_natur[i][j], 3), end=' ')
        print()

    print("\nМатриця Y\n", y)
    print("\nCередні значення функції відгуку за рядками:\n", [round(elem, 3) for elem in y_aver])
    coef(x_natur, y_aver, y, x_norm)

def coef(x, y_aver, y, x_norm):
    skm = lm.LinearRegression(fit_intercept=False)
    skm.fit(x, y_aver)
    b = skm.coef_

    print("\nКоефіцієнти рівняння регресії:")
    b = [round(i, 3) for i in b]
    print(b)
    print("\nРезультат рівняння зі знайденими коефіцієнтами:\n", np.dot(x, b))
    cohren(m, y, y_aver, x_norm, b)

# ----------------- Критерій Кохрена -----------------------
def cohren(m, y, y_aver, x_norm, b):
    print("\nКритерій Кохрена")
    dispersion = []
    for i in range(n):
        z = 0
        for j in range(m):
            z += (y[i][j] - y_aver[i]) ** 2
        dispersion.append(z / m)
    print("Дисперсія:", [round(elem, 3) for elem in dispersion])

    Gp = max(dispersion) / sum(dispersion)
    f1 = m - 1
    f2 = n
    q = 0.05
    Gt = f.ppf(q=(1 - q / f1), dfn=f2, dfd=(f1 - 1) * f2)
    Gt = Gt / (Gt + f1 - 1)
    if Gp < Gt:
        print("Gp < Gt\n{0:.4f} < {1} => дисперсія однорідна".format(Gp, Gt))
        student(m, dispersion, y_aver, x_norm, b)
    else:
        print("Gp > Gt\n{0:.4f} > {1} => дисперсія неоднорідна => m+=1".format(Gp, Gt))
        m += 1
        matrix(m, n)

# ----------------------- Критерій Стюдента --------------------------------
def student(m, dispersion, y_aver, x_norm, b):
    print("\nКритерій Стюдента")
    sb = sum(dispersion) / n
    s_beta = sqrt(sb / (n * m))
    k = len(x_norm[0])
    beta = [sum(y_aver[i] * x_norm[i][j] for i in range(n)) / n for j in range(k)]

    t_t = [abs(beta[i]) / s_beta for i in range(k)]

    f3 = (m - 1) * n
    qq = (1 + 0.95) / 2
    t_table = t.ppf(df=f3, q=qq)

    b_impor = []
    for i in range(k):
        if t_t[i] > t_table:
            b_impor.append(b[i])
        else:
            b_impor.append(0)
    print("Незначні коефіцієнти регресії")
    b_nez = []
    for i in range(k):
        if b[i] not in b_impor:
            b_nez.append(b[i])
            print("b{0} = {1:.3f}".format(i, b[i]))
    global counter
    counter += len(b_nez)
    y_impor = []
    for j in range(n):
        y_impor.append(regression([x_norm[j][i] for i in range(len(t_t))], b_impor))

    print("Значення функції відгуку зі значущими коефіцієнтами\n", [round(elem, 3) for elem in y_impor])
    fisher(m, y_aver, b_impor, y_impor, sb)

# ----------------------- Критерій Фішера --------------------------------
def fisher(m, y_aver, b_impor, y_impor, sb):
    print("\nКритерій Фішера")
    d = 0
    for i in b_impor:
        if i:
            d += 1
    f3 = (m - 1) * n
    f4 = n - d
    s_ad = sum((y_impor[i] - y_aver[i]) ** 2 for i in range(n)) * m / f4
    Fp = s_ad / sb
    Ft = f.ppf(dfn=f4, dfd=f3, q=1 - 0.05)

    if Fp < Ft:
        print("Fp < Ft => {0:.2f} < {1}".format(Fp, Ft))
        print("Отримана математична модель при рівні значимості 0.05 адекватна експериментальним даним")
    else:
        print("Fp > Ft => {0:.2f} > {1}".format(Fp, Ft))
        print("Рівняння регресії неадекватно оригіналу при рівні значимості 0.05")


if __name__ == '__main__':
    n = 15
    m = 3
    i = 0
    while i != 100:
        matrix(m, n)
        i += 1

print("\nЗагальна кількість незначимих коефіцієнтів за 100 ітерацій: ", counter)
