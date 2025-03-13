#   MNC3-4.2.py - Ejercicio 2: Transferencia de calor en una placa metálica
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
    [20, -5, -3, 0, 0],
    [-4, 18, -2, -1, 0],
    [-3, -1, 22, -5, 0],
    [0, -2, -4, 25, -1],
    [0, 0, 0, 0, 1]  # Ajuste para evitar sistema sobredeterminado
])

# Definir el vector de términos independientes
b = np.array([100, 120, 130, 150, 0])

# Resolver el sistema usando Gauss-Seidel
x_sol, errors = gauss_seidel(A, b)

# Guardar errores en un archivo CSV
with open("errors_transferencia_calor.csv", "w", newline="") as f:
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
plt.title("Convergencia del método de Gauss-Seidel para la transferencia de calor")
plt.legend()
plt.grid()
plt.savefig("convergencia_transferencia_calor.png")
plt.show()

# Imprimir solución
print("Solución aproximada para las temperaturas:")
for i, val in enumerate(x_sol, 1):
    print(f"T{i} = {val:.6f}")
