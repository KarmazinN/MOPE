import random
import numpy as np
import sklearn.linear_model as lm
from scipy.stats import f, t
from numpy.linalg import solve


class Lab_4:
    def __init__(self):
        m = 3
        n = 8
        self.x_range = ((-25, -5), (-30, 45), (-5, 5))
        self.y_max = 200 + int(sum([x[1] for x in self.x_range]) / 3)
        self.y_min = 200 + int(sum([x[0] for x in self.x_range]) / 3)
        if not self.linear(n, m):
            self.with_interaction_effect(n, m)

    def regression(self, x, b):
        y = sum([x[i] * b[i] for i in range(len(x))])
        return y

    def dispersion(self, y, y_aver, n, m):
        res = []
        for i in range(n):
            s = sum([(y_aver[i] - y[i][j]) ** 2 for j in range(m)]) / m
            res.append(round(s, 3))
        return res

    def planing_matrix_interaction_effect(self, n, m):
        x_normalized = [[1, -1, -1, -1],
                        [1, -1, 1, 1],
                        [1, 1, -1, 1],
                        [1, 1, 1, -1],
                        [1, -1, -1, 1],
                        [1, -1, 1, -1],
                        [1, 1, -1, -1],
                        [1, 1, 1, 1]]
        y = np.zeros(shape=(n, m), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                y[i][j] = random.randint(self.y_min, self.y_max)

        for x in x_normalized:
            x.append(x[1] * x[2])
            x.append(x[1] * x[3])
            x.append(x[2] * x[3])
            x.append(x[1] * x[2] * x[3])

        x_normalized = np.array(x_normalized[:len(y)])
        x = np.ones(shape=(len(x_normalized), len(x_normalized[0])), dtype=np.int64)

        for i in range(len(x_normalized)):
            for j in range(1, 4):
                if x_normalized[i][j] == -1:
                    x[i][j] = self.x_range[j - 1][0]
                else:
                    x[i][j] = self.x_range[j - 1][1]

        for i in range(len(x)):
            x[i][4] = x[i][1] * x[i][2]
            x[i][5] = x[i][1] * x[i][3]
            x[i][6] = x[i][2] * x[i][3]
            x[i][7] = x[i][1] * x[i][3] * x[i][2]

        print(f'\nМатриця планування для n = {n}, m = {m}:')
        print('\nЗ кодованими значеннями факторів:')
        print('     X0    X1    X2    X3  X1X2  X1X3  X2X3 X1X2X3   Y1    Y2     Y3')
        print(np.concatenate((x, y), axis=1))
        print('\nНормовані значення факторів:\n')
        print(x_normalized)

        return x, y, x_normalized


    def find_coef(self, X, Y, norm=False):
        skm = lm.LinearRegression(fit_intercept=False)
        skm.fit(X, Y)
        B = skm.coef_

        if norm == 1:
            print('\nКоефіцієнти рівняння регресії з нормованими X:')
        else:
            print('\nКоефіцієнти рівняння регресії:')
        B = [round(i, 3) for i in B]
        print(B)
        return B


    def bs(self, x, y, y_aver, n):
        res = [sum(1 * y for y in y_aver) / n]
        for i in range(7):
            b = sum(j[0] * j[1] for j in zip(x[:, i], y_aver)) / n
            res.append(b)
        return res


    def kriteriy_studenta2(self, x, y, y_aver, n, m):
        S_kv = self.dispersion(y, y_aver, n, m)
        s_kv_aver = sum(S_kv) / n
        s_Bs = (s_kv_aver / n / m) ** 0.5
        Bs = self.bs(x, y, y_aver, n)
        ts = [round(abs(B) / s_Bs, 3) for B in Bs]

        return ts


    def kriteriy_studenta(self, x, y_average, n, m, dispersion):
        dispersion_average = sum(dispersion) / n
        s_beta_s = (dispersion_average / n / m) ** 0.5

        beta = [sum(1 * y for y in y_average) / n]
        for i in range(3):
            b = sum(j[0] * j[1] for j in zip(x[:,i], y_average)) / n
            beta.append(b)

        t = [round(abs(b) / s_beta_s, 3) for b in beta]

        return t


    def kriteriy_fishera(self, y, y_average, y_new, n, m, d, dispersion):
        S_ad = m / (n - d) * sum([(y_new[i] - y_average[i])**2 for i in range(len(y))])
        dispersion_average = sum(dispersion) / n

        return S_ad / dispersion_average


    def check(self, X, Y, B, n, m, norm=False):

        f1 = m - 1
        f2 = n
        f3 = f1 * f2
        q = 0.05

        y_aver = [round(sum(i) / len(i), 3) for i in Y]
        print('\nСереднє значення y:', y_aver)

        dispersion_arr = self.dispersion(Y, y_aver, n, m)

        qq = (1 + 0.95) / 2
        student_cr_table = t.ppf(df=f3, q=qq)

        ts = self.kriteriy_studenta2(X[:, 1:], Y, y_aver, n, m)

        temp_cohren = f.ppf(q=(1 - q / f1), dfn=f2, dfd=(f1 - 1) * f2)
        cohren_cr_table = temp_cohren / (temp_cohren + f1 - 1)
        Gp = max(dispersion_arr) / sum(dispersion_arr)

        print('Дисперсія y:', dispersion_arr)

        print(f'Gp = {Gp}')
        if Gp < cohren_cr_table:
            print(f'  З ймовірністю {1-q} дисперсії однорідні.')
        else:
            print("Необхідно збільшити кількість дослідів")
            m += 1
            self.with_interaction_effect(n, m)

        print('\nКритерій Стьюдента:\n', ts)
        res = [t for t in ts if t > student_cr_table]
        final_k = [B[i] for i in range(len(ts)) if ts[i] in res]
        print('\nКоефіцієнти {} статистично незначущі, тому ми виключаємо їх з рівняння.'.format(
            [round(i, 3) for i in B if i not in final_k]))

        y_new = []
        for j in range(n):
            y_new.append(self.regression([X[j][i] for i in range(len(ts)) if ts[i] in res], final_k))

        print(f'\nЗначення "y" з коефіцієнтами {final_k}')
        print(y_new)

        d = len(res)
        if d >= n:
            print('\nF4 <= 0')
            print('')
            return
        f4 = n - d

        Fp = self.kriteriy_fishera(Y, y_aver, y_new, n, m, d, dispersion_arr)

        Ft = f.ppf(dfn=f4, dfd=f3, q=1 - 0.05)

        print('\nПеревірка адекватності за критерієм Фішера')
        print('Fp =', Fp)
        print('Ft =', Ft)
        if Fp < Ft:
            print('Математична модель адекватна експериментальним даним')
            return True
        else:
            print('Математична модель не адекватна експериментальним даним')
            return False

    def with_interaction_effect(self, n, m):
        X, Y, X_norm = self.planing_matrix_interaction_effect(n, m)

        y_aver = [round(sum(i) / len(i), 3) for i in Y]

        B_norm = self.find_coef(X_norm, y_aver, norm=True)

        return self.check(X_norm, Y, B_norm, n, m, norm=True)

    def planning_matrix_linear(self, n, m, x_range):
        x_normalized = np.array([[1, -1, -1, -1],
                                 [1, -1, 1, 1],
                                 [1, 1, -1, 1],
                                 [1, 1, 1, -1],
                                 [1, -1, -1, 1],
                                 [1, -1, 1, -1],
                                 [1, 1, -1, -1],
                                 [1, 1, 1, 1]])
        y = np.zeros(shape=(n,m))
        for i in range(n):
            for j in range(m):
                y[i][j] = random.randint(self.y_min, self.y_max)

        x_normalized = x_normalized[:len(y)]

        x = np.ones(shape=(len(x_normalized), len(x_normalized[0])))
        for i in range(len(x_normalized)):
            for j in range(1, len(x_normalized[i])):
                if x_normalized[i][j] == -1:
                    x[i][j] = x_range[j-1][0]
                else:
                    x[i][j] = x_range[j-1][1]

        print('Матриця планування:' )
        print('    X0  X1   X2   X3   Y1   Y2   Y3  ')
        print(np.concatenate((x, y), axis=1))

        return x, y, x_normalized

    def regression_equation(self, x, y, n):
        y_average = [round(sum(i) / len(i), 2) for i in y]

        mx1 = sum(x[:, 1]) / n
        mx2 = sum(x[:, 2]) / n
        mx3 = sum(x[:, 3]) / n

        my = sum(y_average) / n

        a1 = sum([y_average[i] * x[i][1] for i in range(len(x))]) / n
        a2 = sum([y_average[i] * x[i][2] for i in range(len(x))]) / n
        a3 = sum([y_average[i] * x[i][3] for i in range(len(x))]) / n

        a12 = sum([x[i][1] * x[i][2] for i in range(len(x))]) / n
        a13 = sum([x[i][1] * x[i][3] for i in range(len(x))]) / n
        a23 = sum([x[i][2] * x[i][3] for i in range(len(x))]) / n

        a11 = sum([i ** 2 for i in x[:, 1]]) / n
        a22 = sum([i ** 2 for i in x[:, 2]]) / n
        a33 = sum([i ** 2 for i in x[:, 3]]) / n

        X = [[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a23], [mx3, a13, a23, a33]]
        Y = [my, a1, a2, a3]
        B = [round(i, 2) for i in solve(X, Y)]

        print('\nРівняння регресії:')
        print(f'y = {B[0]} + {B[1]}*x1 + {B[2]}*x2 + {B[3]}*x3')

        return y_average, B


    def linear(self, n, m):
        f1 = m - 1
        f2 = n
        f3 = f1 * f2
        q = 0.05

        x, y, x_norm = self.planning_matrix_linear(n, m, self.x_range)

        y_average, B = self.regression_equation(x, y, n)

        dispersion_arr = self.dispersion(y, y_average, n, m)

        temp_cohren = f.ppf(q=(1 - q / f1), dfn=f2, dfd=(f1 - 1) * f2)
        cohren_cr_table = temp_cohren / (temp_cohren + f1 - 1)
        Gp = max(dispersion_arr) / sum(dispersion_arr)

        print('\nПеревірка за критерієм Кохрена:')
        print(f'  Розрахункове значення: Gp = {Gp}'
              f'\n  Табличне значення: Gt = {cohren_cr_table}')
        if Gp < cohren_cr_table:
            print(f'З ймовірністю {1-q} дисперсії однорідні.')
        else:
            print("Необхідно збільшити ксть дослідів")
            m += 1
            self.linear(n, m)

        qq = (1 + 0.95) / 2
        student_cr_table = t.ppf(df=f3, q=qq)
        student_t = self.kriteriy_studenta(x_norm[:,1:], y_average, n, m, dispersion_arr)

        print('\nТабличне значення критерій Стьюдента:\n ', student_cr_table)
        print('\nРозрахункове значення критерій Стьюдента:\n ', student_t)
        res_student_t = [temp for temp in student_t if temp > student_cr_table]
        final_coefficients = [B[student_t.index(i)] for i in student_t if i in res_student_t]
        print('\nКоефіцієнти {} статистично незначущі.'.
              format([i for i in B if i not in final_coefficients]))

        y_new = []
        for j in range(n):
            y_new.append(self.regression([x[j][student_t.index(i)] for i in student_t if i in res_student_t], final_coefficients))

        print(f'\nОтримаємо значення рівння регресії для {m} дослідів: ')
        print(" ", y_new)

        d = len(res_student_t)
        f4 = n - d
        Fp = self.kriteriy_fishera(y, y_average, y_new, n, m, d, dispersion_arr)
        Ft = f.ppf(dfn=f4, dfd=f3, q=1 - 0.05)

        print('\nПеревірка адекватності за критерієм Фішера:')
        print('  Розрахункове значення критерія Фішера: Fp =', Fp)
        print('  Табличне значення критерія Фішера: Ft =', Ft)
        if Fp < Ft:
            print('\nМатематична модель адекватна експериментальним даним')
            return True
        else:
            print('\nМатематична модель не адекватна експериментальним даним')
            return False


Lab_4()
