#   MNC3-4.1.py - Ejercicio 1: Análisis de un circuito eléctrico
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
    [10, 2, 3, 1],
    [2, 12, 2, 3],
    [3, 2, 15, 1],
    [1, 3, 1, 10]
])

# Definir el vector de términos independientes
b = np.array([15, 22, 18, 10])

# Resolver el sistema usando Gauss-Seidel
x_sol, errors = gauss_seidel(A, b)

# Guardar errores en un archivo CSV
with open("errors_circuito.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Iteración", "Error absoluto", "Error relativo", "Error cuadrático"])
    writer.writerows(errors)
    writer.writerow([])  # Salto de línea
    writer.writerow(["Solución aproximada"])
    for val in x_sol:
        writer.writerow([val])

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
plt.title("Convergencia del método de Gauss-Seidel para el circuito eléctrico")
plt.legend()
plt.grid()
plt.savefig("convergencia_circuito.png")
plt.show()

# Imprimir solución
print("Solución aproximada para las corrientes:")
for i, val in enumerate(x_sol, 1):
    print(f"I{i} = {val:.6f}")
