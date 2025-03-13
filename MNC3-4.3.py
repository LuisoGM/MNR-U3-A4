#   MNC3-4.3.py - Ejercicio 3: Modelo de economía lineal
#   Modificaciones realizadas por: Luis Jorge Fuentes Tec

#   Codigo que implementa el esquema numerico 
#   del metodo iterativo de Gauss-Seidel para
#   resolver sistemas de ecuaciones

#           Autor:
#   Dr. Ivan de Jesus May-Cen
#   imaycen@hotmail.com
#   Version 1.0 : 28/02/2025

#Importacion de: numpy, matplotlib.pyplot y csv
import numpy as np
import matplotlib.pyplot as plt
import csv

# Implementar el método de Gauss-Seidel
def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n)
    x_prev = np.copy(x)
    errors = []
    
    # Iterar hasta...
    for k in range(max_iter):
        # Calcular las nuevas aproximaciones
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_prev[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum1 - sum2) / A[i][i]
        
        # Calcular errores
        abs_error = np.linalg.norm(x - x_prev, ord=np.inf)
        rel_error = abs_error / (np.linalg.norm(x, ord=np.inf) + 1e-10)
        quad_error = np.linalg.norm(x - x_prev) ** 2
        
        errors.append((k, abs_error, rel_error, quad_error))
        
        if abs_error < tol:
            break
        
        x_prev = np.copy(x)
    
    return x, errors

# Definir la matriz de coeficientes y el vector de términos independientes
A = np.array([
    [15, -4, -1, -2,  0,  0,  0,  0,  0,  0],
    [-3, 18, -2,  0, -1,  0,  0,  0,  0,  0],
    [-1, -2, 20,  0,  0, -5,  0,  0,  0,  0],
    [-2, -1, -4, 22,  0,  0, -1,  0,  0,  0],
    [ 0, -1, -3, -1, 25,  0,  0, -2,  0,  0],
    [ 0,  0, -2,  0, -1, 28,  0,  0, -1,  0],
    [ 0,  0,  0, -4,  0, -2, 30,  0,  0, -3],
    [ 0,  0,  0,  0, -1,  0, -1, 35, -2,  0],
    [ 0,  0,  0,  0,  0, -2,  0, -3, 40, -1],
    [ 0,  0,  0,  0,  0,  0, -3,  0, -1, 45]
])

# Definir el vector de términos independientes
b = np.array([200, 250, 180, 300, 270, 310, 320, 400, 450, 500])

# Resolver el sistema usando Gauss-Seidel
x_sol, errors = gauss_seidel(A, b)

# Guardar errores en un archivo CSV
with open("errors_economia.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Iteración", "Error absoluto", "Error relativo", "Error cuadrático"])
    writer.writerows(errors)
    writer.writerow([])
    writer.writerow(["Solución aproximada"])
    for i, val in enumerate(x_sol, start=1):
        writer.writerow([f"x{i}", val])

# Graficar errores
iterations = [e[0] for e in errors]
abs_errors = [e[1] for e in errors]
rel_errors = [e[2] for e in errors]
quad_errors = [e[3] for e in errors]

# Graficar errores
plt.figure(figsize=(10, 5))
plt.plot(iterations, abs_errors, label="Error absoluto")
plt.plot(iterations, rel_errors, label="Error relativo")
plt.plot(iterations, quad_errors, label="Error cuadrático")
plt.yscale("log")
plt.xlabel("Iteraciones")
plt.ylabel("Errores")
plt.title("Convergencia del método de Gauss-Seidel - Economía Lineal")
plt.legend()
plt.grid()
plt.savefig("convergencia_gauss_seidel_economia.png")
plt.show()

# Imprimir solución
print("Valores de x1, x2, ..., x10:")
for i, val in enumerate(x_sol, start=1):
    print(f"x{i} = {val}")
