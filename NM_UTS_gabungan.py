
##### Task 1: Linear Model #####

import numpy as np
import matplotlib.pyplot as plt

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                     10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                     20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                     30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                     40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                     50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                     60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                     70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                     80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                     90, 91, 92, 93, 94, 95, 96, 97, 98, 99
                    ])
Y_data = np.array([
    0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
    7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
    9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
    16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
    25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
    37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
    55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
    71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
    95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
    113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
    141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
    172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
    200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
    233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
    265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
    304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
    343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
    388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
    434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
    479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747
])

# Inisialisasi
n_data = len(X_data)
sum_x = np.sum(X_data)
sum_y = np.sum(Y_data)
sum_xy = np.sum(X_data * Y_data)
sum_x_squared = np.sum(X_data ** 2)

# Slope dan intercept
m = (n_data * sum_xy - sum_x * sum_y) / (n_data * sum_x_squared - (sum_x)**2)
b = (sum_y - m * sum_x) / n_data

# Prediksi y setelah waktu x
Y_prediksi = m * X_data + b

# Standard deviation (Sy)
S_y = np.sqrt(np.sum((Y_data - np.mean(Y_data)) ** 2) / (n_data - 1))

# Standard error (Sy/x)
S_y_x = np.sqrt(np.sum((Y_data - Y_prediksi) ** 2) / (n_data - 2))

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot actual data
plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)

# Plot linear regression
plt.plot(X_data, Y_prediksi, color='pink', label='Linear Regression', linewidth=1.5)

# Adding labels and title
plt.xlabel('Minute')
plt.ylabel('Output (kW)')
plt.title('PV System Power Output')

# Show legend
plt.legend()

# Show the plot
plt.show()

# Print slope, intercept, standard deviation, and standard error
print(f"Slope (Linear Regression): {m}")
print(f"Intercept (Linear Regression): {b}")
print(f"Standard Deviation (Sy): {S_y}")
print(f"Standard Error (Sy/x): {S_y_x}")





 ##### Task 2: Polynomial Model #####
   ##### Gauss Jordan Method #####

import numpy as np
import matplotlib.pyplot as plt
import time

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99], dtype=np.float64)

Y_data = np.array([0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
                   7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
                   9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
                   16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
                   25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
                   37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
                   55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
                   71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
                   95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
                   113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
                   141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
                   172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
                   200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
                   233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
                   265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
                   304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
                   343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
                   388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
                   434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
                   479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747], dtype=np.float64)

# Gauss-Jordan Elimination method
def gauss_jordan_elimination(A, b):
    n = len(b)
    
    # Forward elimination and partial pivoting
    for i in range(n):
        if A[i, i] == 0:
            raise ValueError("Division by zero detected!")
        # Normalize the current row
        factor = A[i, i]
        A[i, :] = A[i, :] / factor
        b[i] = b[i] / factor
        
        # Eliminate all other entries in column i
        for j in range(n):
            if i != j:
                factor = A[j, i]
                A[j, :] -= factor * A[i, :]
                b[j] -= factor * b[i]
    
    return b

# Construct Vandermonde matrix
def construct_vandermonde(X, order):
    return np.vstack([X**i for i in range(order + 1)]).T

# Solve polynomial coefficients using Gauss-Jordan Elimination
def solve_polynomial(X, Y, order):
    A = construct_vandermonde(X, order)
    A_T_A = np.dot(A.T, A)
    A_T_Y = np.dot(A.T, Y)
    
    coefficients = gauss_jordan_elimination(np.copy(A_T_A), np.copy(A_T_Y))
    return coefficients

# Compute R-squared
def compute_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Timing, accuracy, and convergence evaluation
orders = [2, 3, 4, 5]
r_squared_values = []
times_taken = []
convergence_status = []

for order in orders:
    try:
        # Start timing
        start_time = time.time()
        
        # Solve using Gauss-Jordan Elimination
        coeffs = solve_polynomial(X_data, Y_data, order)
        
        # End timing
        time_taken = time.time() - start_time
        times_taken.append(time_taken)
        
        # Predicted values
        Y_poly_pred = np.polyval(coeffs[::-1], X_data)  # Reverse coefficients for np.polyval
        
        # Compute R-squared
        r_squared = compute_r_squared(Y_data, Y_poly_pred)
        r_squared_values.append(r_squared)

        # If no errors, convergence is successful
        convergence_status.append("Converged")

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)
        plt.plot(X_data, Y_poly_pred, label=f'Polynomial Order {order}', color='pink', linewidth=1.5)
        plt.xlabel('Minute')
        plt.ylabel('Output (kW)')
        plt.title(f'Polynomial Regression (Order {order})')
        plt.legend()
        plt.show()

        print(f"Order {order}: Coefficients = {coeffs}")
        print(f"Order {order}: R-squared = {r_squared}")
        print(f"Order {order}: Time taken = {time_taken:.6f} seconds")
        
    except Exception as e:
        convergence_status.append(f"Failed to converge: {str(e)}")
        times_taken.append(None)
        r_squared_values.append(None)
        print(f"Order {order} failed to converge due to: {e}")

# Print summary of results
print("\nSummary of Results:")
for order, r_squared, time_taken, status in zip(orders, r_squared_values, times_taken, convergence_status):
    if r_squared is not None:
        print(f"Polynomial Order {order}: R-squared = {r_squared:.6f}, Time taken = {time_taken:.6f} seconds, Status: {status}")
    else:
        print(f"Polynomial Order {order}: Status: {status}")
        
        
        
        

 ##### Task 2: Polynomial Model #####
       ##### Gauss Seidel #####
       
import numpy as np
import matplotlib.pyplot as plt
import time

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99], dtype=np.float64)

Y_data = np.array([0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
                   7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
                   9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
                   16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
                   25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
                   37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
                   55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
                   71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
                   95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
                   113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
                   141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
                   172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
                   200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
                   233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
                   265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
                   304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
                   343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
                   388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
                   434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
                   479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747], dtype=np.float64)

# Gauss-Seidel Method
def gauss_seidel(A, b, tol=1e-10, max_iterations=100000000):
    n = len(b)
    x = np.zeros_like(b, dtype=np.float64)
    for k in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.allclose(x, x_new, tol):
            return x_new, k + 1
        x = x_new
    raise ValueError("Gauss-Seidel method did not converge within the maximum number of iterations.")

# Construct Vandermonde matrix
def construct_vandermonde(X, order):
    return np.vstack([X**i for i in range(order + 1)]).T

# Solve polynomial coefficients using Gauss-Seidel Method
def solve_polynomial_gauss_seidel(X, Y, order):
    A = construct_vandermonde(X, order)
    A_T_A = np.dot(A.T, A)
    A_T_Y = np.dot(A.T, Y)
    
    # Solve the system using Gauss-Seidel
    coefficients, iterations = gauss_seidel(np.copy(A_T_A), np.copy(A_T_Y))
    return coefficients, iterations

# Compute R-squared
def compute_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Timing, accuracy, and convergence evaluation
orders = [2, 3, 4, 5]
r_squared_values = []
times_taken = []
iterations_needed = []
convergence_status = []

for order in orders:
    try:
        # Start timing
        start_time = time.time()
        
        # Solve using Gauss-Seidel
        coeffs, iterations = solve_polynomial_gauss_seidel(X_data, Y_data, order)
        
        # End timing
        time_taken = time.time() - start_time
        times_taken.append(time_taken)
        iterations_needed.append(iterations)
        
        # Predicted values
        Y_poly_pred = np.polyval(coeffs[::-1], X_data)  # Reverse coefficients for np.polyval
        
        # Compute R-squared
        r_squared = compute_r_squared(Y_data, Y_poly_pred)
        r_squared_values.append(r_squared)

        # If no errors, convergence is successful
        convergence_status.append("Converged")

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)
        plt.plot(X_data, Y_poly_pred, label=f'Polynomial Order {order}', color='pink', linewidth=1.5)
        plt.xlabel('Minute')
        plt.ylabel('Output (kW)')
        plt.title(f'Polynomial Regression (Order {order})')
        plt.legend()
        plt.show()

        print(f"Order {order}: Coefficients = {coeffs}")
        print(f"Order {order}: R-squared = {r_squared}")
        print(f"Order {order}: Time taken = {time_taken:.6f} seconds")
        print(f"Order {order}: Iterations needed = {iterations}")

    except Exception as e:
        convergence_status.append(f"Failed to converge: {str(e)}")
        times_taken.append(None)
        r_squared_values.append(None)
        iterations_needed.append(None)
        print(f"Order {order} failed to converge due to: {e}")

# Print summary of results
print("\nSummary of Results:")
for order, r_squared, time_taken, iterations, status in zip(orders, r_squared_values, times_taken, iterations_needed, convergence_status):
    if r_squared is not None:
        print(f"Polynomial Order {order}: R-squared = {r_squared:.6f}, Time taken = {time_taken:.6f} seconds, Iterations: {iterations}, Status: {status}")
    else:
        print(f"Polynomial Order {order}: Status: {status}")
        
        
        


 ##### Task 2: Polynomial Model #####
    ##### Jacobi Iteration #####
    
import numpy as np
import matplotlib.pyplot as plt
import time

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99], dtype=np.float64)

Y_data = np.array([0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
                   7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
                   9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
                   16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
                   25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
                   37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
                   55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
                   71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
                   95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
                   113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
                   141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
                   172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
                   200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
                   233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
                   265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
                   304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
                   343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
                   388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
                   434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
                   479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747], dtype=np.float64)

# LU Decomposition (Doolittle's Method)
def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    # Decompose A into L and U
    for i in range(n):
        for j in range(i, n):
            # Upper triangular U
            sum_u = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - sum_u
        
        for j in range(i, n):
            if i == j:
                L[i][i] = 1  # Diagonal as 1
            else:
                sum_l = sum(L[j][k] * U[k][i] for k in range(i))
                L[j][i] = (A[j][i] - sum_l) / U[i][i]
    
    return L, U

# Forward substitution to solve Ly = b
def forward_substitution(L, b):
    n = len(b)
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

# Backward substitution to solve Ux = y
def backward_substitution(U, y):
    n = len(y)
    x = np.zeros_like(y)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

# Construct Vandermonde matrix
def construct_vandermonde(X, order):
    return np.vstack([X**i for i in range(order + 1)]).T

# Solve polynomial coefficients using LU Decomposition
def solve_polynomial(X, Y, order):
    A = construct_vandermonde(X, order)
    A_T_A = np.dot(A.T, A)
    A_T_Y = np.dot(A.T, Y)
    
    # LU Decomposition
    L, U = lu_decomposition(A_T_A)
    
    # Forward and backward substitution
    y = forward_substitution(L, A_T_Y)
    coefficients = backward_substitution(U, y)
    
    return coefficients

# Compute R-squared
def compute_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Timing, accuracy, and convergence evaluation
orders = [2, 3, 4, 5]
r_squared_values = []
times_taken = []
convergence_status = []

for order in orders:
    try:
        # Start timing
        start_time = time.time()
        
        # Solve using LU Decomposition
        coeffs = solve_polynomial(X_data, Y_data, order)
        
        # End timing
        time_taken = time.time() - start_time
        times_taken.append(time_taken)
        
        # Predicted values
        Y_poly_pred = np.polyval(coeffs[::-1], X_data)  # Reverse coefficients for np.polyval
        
        # Compute R-squared
        r_squared = compute_r_squared(Y_data, Y_poly_pred)
        r_squared_values.append(r_squared)

        # If no errors, convergence is successful
        convergence_status.append("Converged")

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)
        plt.plot(X_data, Y_poly_pred, label=f'Polynomial Order {order}', color='pink', linewidth=1.5)
        plt.xlabel('Minute')
        plt.ylabel('Output (kW)')
        plt.title(f'Polynomial Regression (Order {order})')
        plt.legend()
        plt.show()

        print(f"Order {order}: Coefficients = {coeffs}")
        print(f"Order {order}: R-squared = {r_squared}")
        print(f"Order {order}: Time taken = {time_taken:.6f} seconds")
        
    except Exception as e:
        convergence_status.append(f"Failed to converge: {str(e)}")
        times_taken.append(None)
        r_squared_values.append(None)
        print(f"Order {order} failed to converge due to: {e}")

# Print summary of results
print("\nSummary of Results:")
for order, r_squared, time_taken, status in zip(orders, r_squared_values, times_taken, convergence_status):
    if r_squared is not None:
        print(f"Polynomial Order {order}: R-squared = {r_squared:.6f}, Time taken = {time_taken:.6f} seconds, Status: {status}")
    else:
        print(f"Polynomial Order {order}: Status: {status}")
        
        
        
        
        

 ##### Task 2: Polynomial Model #####
    ##### LU Decomposition #####
    
import numpy as np
import matplotlib.pyplot as plt
import time

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99], dtype=np.float64)

Y_data = np.array([0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
                   7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
                   9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
                   16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
                   25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
                   37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
                   55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
                   71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
                   95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
                   113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
                   141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
                   172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
                   200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
                   233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
                   265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
                   304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
                   343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
                   388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
                   434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
                   479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747], dtype=np.float64)

# LU Decomposition (Doolittle's Method)
def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    # Decompose A into L and U
    for i in range(n):
        for j in range(i, n):
            # Upper triangular U
            sum_u = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - sum_u
        
        for j in range(i, n):
            if i == j:
                L[i][i] = 1  # Diagonal as 1
            else:
                sum_l = sum(L[j][k] * U[k][i] for k in range(i))
                L[j][i] = (A[j][i] - sum_l) / U[i][i]
    
    return L, U

# Forward substitution to solve Ly = b
def forward_substitution(L, b):
    n = len(b)
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

# Backward substitution to solve Ux = y
def backward_substitution(U, y):
    n = len(y)
    x = np.zeros_like(y)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

# Construct Vandermonde matrix
def construct_vandermonde(X, order):
    return np.vstack([X**i for i in range(order + 1)]).T

# Solve polynomial coefficients using LU Decomposition
def solve_polynomial(X, Y, order):
    A = construct_vandermonde(X, order)
    A_T_A = np.dot(A.T, A)
    A_T_Y = np.dot(A.T, Y)
    
    # LU Decomposition
    L, U = lu_decomposition(A_T_A)
    
    # Forward and backward substitution
    y = forward_substitution(L, A_T_Y)
    coefficients = backward_substitution(U, y)
    
    return coefficients

# Compute R-squared
def compute_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Timing, accuracy, and convergence evaluation
orders = [2, 3, 4, 5]
r_squared_values = []
times_taken = []
convergence_status = []

for order in orders:
    try:
        # Start timing
        start_time = time.time()
        
        # Solve using LU Decomposition
        coeffs = solve_polynomial(X_data, Y_data, order)
        
        # End timing
        time_taken = time.time() - start_time
        times_taken.append(time_taken)
        
        # Predicted values
        Y_poly_pred = np.polyval(coeffs[::-1], X_data)  # Reverse coefficients for np.polyval
        
        # Compute R-squared
        r_squared = compute_r_squared(Y_data, Y_poly_pred)
        r_squared_values.append(r_squared)

        # If no errors, convergence is successful
        convergence_status.append("Converged")

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)
        plt.plot(X_data, Y_poly_pred, label=f'Polynomial Order {order}', color='pink', linewidth=1.5)
        plt.xlabel('Minute')
        plt.ylabel('Output (kW)')
        plt.title(f'Polynomial Regression (Order {order})')
        plt.legend()
        plt.show()

        print(f"Order {order}: Coefficients = {coeffs}")
        print(f"Order {order}: R-squared = {r_squared}")
        print(f"Order {order}: Time taken = {time_taken:.6f} seconds")
        
    except Exception as e:
        convergence_status.append(f"Failed to converge: {str(e)}")
        times_taken.append(None)
        r_squared_values.append(None)
        print(f"Order {order} failed to converge due to: {e}")

# Print summary of results
print("\nSummary of Results:")
for order, r_squared, time_taken, status in zip(orders, r_squared_values, times_taken, convergence_status):
    if r_squared is not None:
        print(f"Polynomial Order {order}: R-squared = {r_squared:.6f}, Time taken = {time_taken:.6f} seconds, Status: {status}")
    else:
        print(f"Polynomial Order {order}: Status: {status}")
        
        
        
        
        

 ##### Task 2: Polynomial Model #####
##### Naive Gaussian Elimination #####

import numpy as np
import matplotlib.pyplot as plt
import time

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99], dtype=np.float64)

Y_data = np.array([
    0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
    7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
    9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
    16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
    25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
    37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
    55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
    71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
    95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
    113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
    141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
    172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
    200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
    233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
    265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
    304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
    343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
    388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
    434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
    479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747], dtype=np.float64)

# Naive Gaussian Elimination
def naive_gaussian_elimination(A, b):
    n = len(b)
    
    # Forward elimination
    for i in range(n):
        for j in range(i+1, n):
            if A[i, i] == 0:
                raise ValueError("Division by zero detected!")
            factor = A[j, i] / A[i, i]
            for k in range(i, n):
                A[j, k] -= factor * A[i, k]
            b[j] -= factor * b[i]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum_ax = 0
        for j in range(i+1, n):
            sum_ax += A[i, j] * x[j]
        x[i] = (b[i] - sum_ax) / A[i, i]
    
    return x

# Construct Vandermonde matrix
def construct_vandermonde(X, order):
    return np.vstack([X**i for i in range(order + 1)]).T

# Solve polynomial coefficients using Naive Gaussian Elimination
def solve_polynomial(X, Y, order):
    A = construct_vandermonde(X, order)
    A_T_A = np.dot(A.T, A)
    A_T_Y = np.dot(A.T, Y)
    
    coefficients = naive_gaussian_elimination(np.copy(A_T_A), np.copy(A_T_Y))
    return coefficients

# Compute R-squared
def compute_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Timing, accuracy, and convergence evaluation
orders = [2, 3, 4, 5]
r_squared_values = []
times_taken = []
convergence_status = []

for order in orders:
    try:
        # Start timing
        start_time = time.time()
        
        # Solve using Naive Gaussian Elimination
        coeffs = solve_polynomial(X_data, Y_data, order)
        
        # End timing
        time_taken = time.time() - start_time
        times_taken.append(time_taken)
        
        # Predicted values
        Y_poly_pred = np.polyval(coeffs[::-1], X_data)  # Reverse coefficients for np.polyval
        
        # Compute R-squared
        r_squared = compute_r_squared(Y_data, Y_poly_pred)
        r_squared_values.append(r_squared)

        # If no errors, convergence is successful
        convergence_status.append("Converged")

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)
        plt.plot(X_data, Y_poly_pred, label=f'Polynomial Order {order}', color='pink', linewidth=1.5)
        plt.xlabel('Minute')
        plt.ylabel('Output (kW)')
        plt.title(f'Polynomial Regression (Order {order})')
        plt.legend()
        plt.show()

        print(f"Order {order}: Coefficients = {coeffs}")
        print(f"Order {order}: R-squared = {r_squared}")
        print(f"Order {order}: Time taken = {time_taken:.6f} seconds")
        
    except Exception as e:
        convergence_status.append(f"Failed to converge: {str(e)}")
        times_taken.append(None)
        r_squared_values.append(None)
        print(f"Order {order} failed to converge due to: {e}")

# Print summary of results
print("\nSummary of Results:")
for order, r_squared, time_taken, status in zip(orders, r_squared_values, times_taken, convergence_status):
    if r_squared is not None:
        print(f"Polynomial Order {order}: R-squared = {r_squared:.6f}, Time taken = {time_taken:.6f} seconds, Status: {status}")
    else:
        print(f"Polynomial Order {order}: Status: {status}")
        
        
        
        

         ##### Task 2: Polynomial Model #####
##### Gaussian Elimination with Partial Pivoting #####

import numpy as np
import matplotlib.pyplot as plt
import time

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99], dtype=np.float64)

Y_data = np.array([
    0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
    7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
    9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
    16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
    25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
    37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
    55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
    71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
    95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
    113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
    141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
    172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
    200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
    233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
    265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
    304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
    343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
    388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
    434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
    479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747], dtype=np.float64)

# Gaussian Elimination with Partial Pivoting
def gaussian_elimination_partial_pivoting(A, b):
    n = len(b)
    
    # Forward elimination with partial pivoting
    for i in range(n):
        # Pivot: Find the max element in the column and swap rows
        max_row = np.argmax(np.abs(A[i:, i])) + i
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]
        
        for j in range(i+1, n):
            if A[i, i] == 0:
                raise ValueError("Division by zero detected!")
            factor = A[j, i] / A[i, i]
            for k in range(i, n):
                A[j, k] -= factor * A[i, k]
            b[j] -= factor * b[i]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum_ax = 0
        for j in range(i+1, n):
            sum_ax += A[i, j] * x[j]
        x[i] = (b[i] - sum_ax) / A[i, i]
    
    return x

# Construct Vandermonde matrix
def construct_vandermonde(X, order):
    return np.vstack([X**i for i in range(order + 1)]).T

# Solve polynomial coefficients using Gaussian Elimination with Partial Pivoting
def solve_polynomial(X, Y, order):
    A = construct_vandermonde(X, order)
    A_T_A = np.dot(A.T, A)
    A_T_Y = np.dot(A.T, Y)
    
    coefficients = gaussian_elimination_partial_pivoting(np.copy(A_T_A), np.copy(A_T_Y))
    return coefficients

# Compute R-squared
def compute_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Timing, accuracy, and convergence evaluation
orders = [2, 3, 4, 5]
r_squared_values = []
times_taken = []
convergence_status = []

for order in orders:
    try:
        # Start timing
        start_time = time.time()
        
        # Solve using Gaussian Elimination with Partial Pivoting
        coeffs = solve_polynomial(X_data, Y_data, order)
        
        # End timing
        time_taken = time.time() - start_time
        times_taken.append(time_taken)
        
        # Predicted values
        Y_poly_pred = np.polyval(coeffs[::-1], X_data)  # Reverse coefficients for np.polyval
        
        # Compute R-squared
        r_squared = compute_r_squared(Y_data, Y_poly_pred)
        r_squared_values.append(r_squared)

        # If no errors, convergence is successful
        convergence_status.append("Converged")

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)
        plt.plot(X_data, Y_poly_pred, label=f'Polynomial Order {order}', color='pink', linewidth=1.5)
        plt.xlabel('Minute')
        plt.ylabel('Output (kW)')
        plt.title(f'Polynomial Regression (Order {order})')
        plt.legend()
        plt.show()

        print(f"Order {order}: Coefficients = {coeffs}")
        print(f"Order {order}: R-squared = {r_squared}")
        print(f"Order {order}: Time taken = {time_taken:.6f} seconds")
        
    except Exception as e:
        convergence_status.append(f"Failed to converge: {str(e)}")
        times_taken.append(None)
        r_squared_values.append(None)
        print(f"Order {order} failed to converge due to: {e}")

# Print summary of results
print("\nSummary of Results:")
for order, r_squared, time_taken, status in zip(orders, r_squared_values, times_taken, convergence_status):
    if r_squared is not None:
        print(f"Polynomial Order {order}: R-squared = {r_squared:.6f}, Time taken = {time_taken:.6f} seconds, Status: {status}")
    else:
        print(f"Polynomial Order {order}: Status: {status}")
        
        
        
        
        

##### Task 2: Polynomial Model #####

import numpy as np
import matplotlib.pyplot as plt

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

Y_data = np.array([
    0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
    7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
    9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
    16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
    25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
    37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
    55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
    71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
    95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
    113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
    141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
    172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
    200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
    233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
    265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
    304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
    343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
    388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
    434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
    479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747
])

# Function to compute R-squared
def compute_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Function to compute Standard Error
def compute_standard_error(y_true, y_pred, num_params):
    n = len(y_true)  # Number of data points
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    standard_error = np.sqrt(residual_sum_of_squares / (n - num_params))
    return standard_error

# Fit polynomial models and calculate R-squared and Standard Error
orders = [2, 3, 4, 5]
r_squared_values = []
standard_errors = []

for order in orders:
    coeffs = np.polyfit(X_data, Y_data, order)
    Y_poly_pred = np.polyval(coeffs, X_data)
    r_squared = compute_r_squared(Y_data, Y_poly_pred)
    r_squared_values.append(r_squared)
    
    # Compute standard error
    standard_error = compute_standard_error(Y_data, Y_poly_pred, order + 1)  # num_params = order + 1 (including intercept)
    standard_errors.append(standard_error)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)
    plt.plot(X_data, Y_poly_pred, label=f'Polynomial Order {order}', color='pink', linewidth=1.5)
    plt.xlabel('Minute')
    plt.ylabel('Output (kW)')
    plt.title(f'Polynomial Regression (Order {order})')
    plt.legend()
    plt.show()

    print(f"Order {order}: Coefficients = {coeffs}")
    print(f"Order {order}: R-squared = {r_squared}")
    print(f"Order {order}: Standard Error = {standard_error}")

# Print the R-squared values and Standard Errors
for order, r_squared, std_err in zip(orders, r_squared_values, standard_errors):
    print(f"Polynomial Order {order}: R-squared = {r_squared}, Standard Error = {std_err}")
    
    
    
    


##### Task 3: Exponential Model #####
    ##### Bisection Method #####
    
import numpy as np
import matplotlib.pyplot as plt
import time  # To measure computational speed

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

Y_data = np.array([
    0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
    7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
    9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
    16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
    25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
    37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
    55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
    71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
    95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
    113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
    141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
    172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
    200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
    233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
    265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
    304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
    343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
    388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
    434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
    479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747
])

# Logarithmic transformation of Y_data to fit the exponential model
Y_log = np.log(Y_data)

# Perform linear regression on the log-transformed data
coeffs = np.polyfit(X_data, Y_log, 1)

# Exponential model: y = A * exp(B * x)
A = np.exp(coeffs[1])  # exp(intercept)
B = coeffs[0]  # slope

# Predict Y values using the exponential model
Y_exp_pred = A * np.exp(B * X_data)

# Define residual function (difference between actual and predicted values)
def residual_function(x, index):
    return Y_data[index] - (A * np.exp(B * x))

# Bisection method implementation with convergence tracking
def bisection_method(func, a, b, tol=1e-6, max_iterations=100, index=None):
    """
    Bisection method to find the root of a function func in the interval [a, b].
    The function func represents the difference between the actual and predicted data.
    This version tracks the convergence by counting iterations.
    """
    if func(a, index) * func(b, index) >= 0:
        raise ValueError("The function must have opposite signs at a and b.")
    
    iteration = 0
    while (b - a) / 2.0 > tol and iteration < max_iterations:
        c = (a + b) / 2.0
        if func(c, index) == 0:
            return c, iteration  # Return root and number of iterations
        elif func(a, index) * func(c, index) < 0:
            b = c  # Root is in the left half
        else:
            a = c  # Root is in the right half
        iteration += 1
    
    return (a + b) / 2.0, iteration  # Return root and number of iterations

# Choose an index to apply the bisection method (e.g., for the 50th data point)
index_to_find = 50

# Set the initial interval for the bisection method
a_interval = 0
b_interval = 100

# Measure computational speed (time)
start_time = time.time()

# Find the root using the bisection method
root, iterations = bisection_method(residual_function, a_interval, b_interval, tol=1e-6, index=index_to_find)

# End computational time
end_time = time.time()
computational_time = end_time - start_time

# Compute accuracy by evaluating the residual at the root
accuracy = abs(residual_function(root, index_to_find))

# Plot actual data and exponential model
plt.figure(figsize=(10, 6))
plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)
plt.plot(X_data, A * np.exp(B * X_data), color='red', label=f'Exponential Fit: y = {A:.4f} * exp({B:.4f} * x)', linewidth=2)

plt.xlabel('Minute')
plt.ylabel('Output (kW)')
plt.title('Exponential Regression with Bisection Root Calculation')
plt.legend()
plt.show()

# Print the results
print(f"Root for data point {index_to_find} found at x = {root}")
print(f"Convergence: Number of iterations = {iterations}")
print(f"Accuracy (Residual at the root): {accuracy}")
print(f"Computational Speed: {computational_time:.6f} seconds")





##### Task 3: Exponential Model #####
     ##### Brent's Method #####

import numpy as np
import matplotlib.pyplot as plt
import time  # To measure computational speed
from scipy.optimize import brentq

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

Y_data = np.array([
    0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
    7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
    9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
    16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
    25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
    37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
    55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
    71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
    95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
    113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
    141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
    172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
    200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
    233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
    265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
    304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
    343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
    388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
    434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
    479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747
])

# Logarithmic transformation of Y_data to fit the exponential model
Y_log = np.log(Y_data)

# Perform linear regression on the log-transformed data
coeffs = np.polyfit(X_data, Y_log, 1)

# Exponential model: y = A * exp(B * x)
A = np.exp(coeffs[1])  # exp(intercept)
B = coeffs[0]  # slope

# Predict Y values using the exponential model
Y_exp_pred = A * np.exp(B * X_data)

# Define residual function (difference between actual and predicted values)
def residual_function(x, index):
    return Y_data[index] - (A * np.exp(B * x))

# Function to track the number of function evaluations
eval_count = 0  # Global variable to count the number of evaluations

def tracked_residual_function(x, index):
    global eval_count
    eval_count += 1
    return residual_function(x, index)

# Set the initial interval for Brent's method
a_interval = 0
b_interval = 100

# Choose an index to apply Brent's method (e.g., for the 50th data point)
index_to_find = 50

# Measure computational speed (time)
start_time = time.time()

# Apply Brent's method to find the root
root = brentq(tracked_residual_function, a_interval, b_interval, args=(index_to_find,), xtol=1e-6)

# End computational time
end_time = time.time()
computational_time = end_time - start_time

# Compute accuracy by evaluating the residual at the root
accuracy = abs(residual_function(root, index_to_find))

# Plot actual data and exponential model
plt.figure(figsize=(10, 6))
plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)
plt.plot(X_data, A * np.exp(B * X_data), color='red', label=f'Exponential Fit: y = {A:.4f} * exp({B:.4f} * x)', linewidth=2)

plt.xlabel('Minute')
plt.ylabel('Output (kW)')
plt.title('Exponential Regression with Brent\'s Method Root Calculation')
plt.legend()
plt.show()

# Print the results
print(f"Root for data point {index_to_find} found at x = {root}")
print(f"Convergence: Number of function evaluations = {eval_count}")
print(f"Accuracy (Residual at the root): {accuracy}")
print(f"Computational Speed: {computational_time:.6f} seconds")





##### Task 3: Exponential Model #####

import numpy as np
import matplotlib.pyplot as plt

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

Y_data = np.array([
    0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
    7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
    9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
    16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
    25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
    37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
    55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
    71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
    95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
    113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
    141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
    172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
    200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
    233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
    265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
    304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
    343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
    388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
    434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
    479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747
])

# Logarithmic transformation of Y_data to fit the exponential model
Y_log = np.log(Y_data)

# Perform linear regression on the log-transformed data
coeffs = np.polyfit(X_data, Y_log, 1)

# Exponential model: y = A * exp(B * x)
A = np.exp(coeffs[1])  # exp(intercept)
B = coeffs[0]  # slope

# Predict Y values using the exponential model
Y_exp_pred = A * np.exp(B * X_data)

# Plot actual data and exponential model
plt.figure(figsize=(10, 6))
plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)
plt.plot(X_data, Y_exp_pred, color='red', label=f'Exponential Fit: y = {A:.4f} * exp({B:.4f} * x)', linewidth=2)

plt.xlabel('Minute')
plt.ylabel('Output (kW)')
plt.title('Exponential Regression')
plt.legend()
plt.show()

# Function to compute R-squared
def compute_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Function to compute Standard Error
def compute_standard_error(y_true, y_pred, num_params):
    n = len(y_true)  # Number of data points
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    standard_error = np.sqrt(residual_sum_of_squares / (n - num_params))
    return standard_error

# Calculate R-squared
r_squared = compute_r_squared(Y_data, Y_exp_pred)

# Calculate Standard Error
num_params = 2  # Two parameters: A and B
standard_error = compute_standard_error(Y_data, Y_exp_pred, num_params)

# Print the results
print(f"Exponential Model: y = {A:.4f} * exp({B:.4f} * x)")
print(f"R-squared: {r_squared}")
print(f"Standard Error: {standard_error}")





##### Task 3: Exponential Model #####
  ##### False-Position Method #####
  
import numpy as np
import matplotlib.pyplot as plt
import time  # To measure computational speed

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

Y_data = np.array([
    0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
    7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
    9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
    16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
    25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
    37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
    55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
    71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
    95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
    113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
    141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
    172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
    200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
    233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
    265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
    304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
    343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
    388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
    434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
    479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747
])

# Logarithmic transformation of Y_data to fit the exponential model
Y_log = np.log(Y_data)

# Perform linear regression on the log-transformed data
coeffs = np.polyfit(X_data, Y_log, 1)

# Exponential model: y = A * exp(B * x)
A = np.exp(coeffs[1])  # exp(intercept)
B = coeffs[0]  # slope

# Predict Y values using the exponential model
Y_exp_pred = A * np.exp(B * X_data)

# Define residual function (difference between actual and predicted values)
def residual_function(x, index):
    return Y_data[index] - (A * np.exp(B * x))

# False-Position Method implementation with convergence and accuracy tracking
def false_position_method(func, a, b, tol=1e-6, max_iterations=100, index=None):
    """
    False-position method to find the root of a function func in the interval [a, b].
    Tracks convergence and accuracy.
    """
    if func(a, index) * func(b, index) >= 0:
        raise ValueError("The function must have opposite signs at a and b.")
    
    iteration = 0
    while abs(b - a) > tol and iteration < max_iterations:
        # Calculate the false position (linear interpolation)
        c = b - (func(b, index) * (b - a)) / (func(b, index) - func(a, index))
        
        # Check if the root is found
        if abs(func(c, index)) < tol:
            return c, iteration  # Return root and number of iterations
        
        # Update the interval based on the signs of the function
        if func(a, index) * func(c, index) < 0:
            b = c  # Root is in the left half
        else:
            a = c  # Root is in the right half
        
        iteration += 1
    
    return c, iteration  # Return root and number of iterations

# Choose an index to apply the False-Position method (e.g., for the 50th data point)
index_to_find = 50

# Set the initial interval for the False-Position method
a_interval = 0
b_interval = 100

# Measure computational speed (time)
start_time = time.time()

# Find the root using the False-Position method
root, iterations = false_position_method(residual_function, a_interval, b_interval, tol=1e-6, index=index_to_find)

# End computational time
end_time = time.time()
computational_time = end_time - start_time

# Compute accuracy by evaluating the residual at the root
accuracy = abs(residual_function(root, index_to_find))

# Plot actual data and exponential model
plt.figure(figsize=(10, 6))
plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)
plt.plot(X_data, A * np.exp(B * X_data), color='red', label=f'Exponential Fit: y = {A:.4f} * exp({B:.4f} * x)', linewidth=2)

plt.xlabel('Minute')
plt.ylabel('Output (kW)')
plt.title('Exponential Regression with False-Position Method Root Calculation')
plt.legend()
plt.show()

# Print the results
print(f"Root for data point {index_to_find} found at x = {root}")
print(f"Convergence: Number of iterations = {iterations}")
print(f"Accuracy (Residual at the root): {accuracy}")
print(f"Computational Speed: {computational_time:.6f} seconds")





##### Task 3: Exponential Model #####
 ##### Modified Newton-Raphson #####
 
import numpy as np
import matplotlib.pyplot as plt
import time  # To measure computational speed

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

Y_data = np.array([
    0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
    7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
    9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
    16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
    25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
    37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
    55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
    71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
    95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
    113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
    141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
    172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
    200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
    233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
    265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
    304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
    343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
    388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
    434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
    479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747
])

# Logarithmic transformation of Y_data to fit the exponential model
Y_log = np.log(Y_data)

# Perform linear regression on the log-transformed data
coeffs = np.polyfit(X_data, Y_log, 1)

# Exponential model: y = A * exp(B * x)
A = np.exp(coeffs[1])  # exp(intercept)
B = coeffs[0]  # slope

# Predict Y values using the exponential model
Y_exp_pred = A * np.exp(B * X_data)

# Define residual function (difference between actual and predicted values)
def residual_function(x, index):
    return Y_data[index] - (A * np.exp(B * x))

# Derivative of the residual function
def residual_function_derivative(x, index):
    return -A * B * np.exp(B * x)  # Derivative of A * exp(B * x)

# Modified Newton-Raphson Method implementation
def modified_newton_raphson(func, func_prime, x0, m, tol=1e-6, max_iterations=100, index=None):
    """
    Modified Newton-Raphson method to find the root of a function func with its derivative func_prime.
    'm' is the multiplicity of the root.
    """
    iteration = 0
    x = x0
    
    while iteration < max_iterations:
        fx = func(x, index)
        f_prime_x = func_prime(x, index)
        
        # Avoid division by zero
        if abs(f_prime_x) < 1e-12:
            raise ValueError("Derivative near zero; Modified Newton-Raphson method failed.")
        
        # Modified Newton-Raphson update
        x_new = x - m * fx / f_prime_x
        
        # Check convergence
        if abs(x_new - x) < tol:
            return x_new, iteration  # Return the root and number of iterations
        
        x = x_new
        iteration += 1
    
    return x, iteration  # Return root and number of iterations if max iterations are reached

# Choose an index to apply the Modified Newton-Raphson method (e.g., for the 50th data point)
index_to_find = 50

# Initial guess for Modified Newton-Raphson
initial_guess = 50

# Set multiplicity (m) of the root
multiplicity = 1  # Assuming simple root, change this value if the root is known to have a higher multiplicity

# Measure computational speed (time)
start_time = time.time()

# Find the root using Modified Newton-Raphson method
root, iterations = modified_newton_raphson(residual_function, residual_function_derivative, initial_guess, multiplicity, tol=1e-6, index=index_to_find)

# End computational time
end_time = time.time()
computational_time = end_time - start_time

# Compute accuracy by evaluating the residual at the root
accuracy = abs(residual_function(root, index_to_find))

# Plot actual data and exponential model
plt.figure(figsize=(10, 6))
plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)
plt.plot(X_data, A * np.exp(B * X_data), color='red', label=f'Exponential Fit: y = {A:.4f} * exp({B:.4f} * x)', linewidth=2)

plt.xlabel('Minute')
plt.ylabel('Output (kW)')
plt.title('Exponential Regression with Modified Newton-Raphson Method Root Calculation')
plt.legend()
plt.show()

# Print the results
print(f"Root for data point {index_to_find} found at x = {root}")
print(f"Convergence: Number of iterations = {iterations}")
print(f"Accuracy (Residual at the root): {accuracy}")
print(f"Computational Speed: {computational_time:.6f} seconds")





##### Task 3: Exponential Model #####
     ##### Newton's Method #####
     
import numpy as np
import matplotlib.pyplot as plt
import time  # To measure computational speed

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

Y_data = np.array([
    0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
    7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
    9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
    16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
    25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
    37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
    55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
    71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
    95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
    113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
    141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
    172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
    200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
    233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
    265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
    304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
    343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
    388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
    434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
    479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747
])

# Logarithmic transformation of Y_data to fit the exponential model
Y_log = np.log(Y_data)

# Perform linear regression on the log-transformed data
coeffs = np.polyfit(X_data, Y_log, 1)

# Exponential model: y = A * exp(B * x)
A = np.exp(coeffs[1])  # exp(intercept)
B = coeffs[0]  # slope

# Predict Y values using the exponential model
Y_exp_pred = A * np.exp(B * X_data)

# Define residual function (difference between actual and predicted values)
def residual_function(x, index):
    return Y_data[index] - (A * np.exp(B * x))

# Define the derivative of the residual function
def residual_function_derivative(x, index):
    return -A * B * np.exp(B * x)  # Derivative of A * exp(B * x)

# Newton's Method implementation
def newtons_method(func, func_prime, x0, tol=1e-6, max_iterations=100, index=None):
    """
    Newton's method to find the root of a function func with derivative func_prime.
    """
    iteration = 0
    x = x0
    
    while iteration < max_iterations:
        fx = func(x, index)
        f_prime_x = func_prime(x, index)
        
        # Avoid division by zero
        if abs(f_prime_x) < 1e-12:
            raise ValueError("Derivative too small; Newton's method may not converge.")
        
        # Newton's method formula
        x_new = x - fx / f_prime_x
        
        # Check for convergence
        if abs(x_new - x) < tol:
            return x_new, iteration  # Return the root and number of iterations
        
        x = x_new
        iteration += 1
    
    return x, iteration  # Return root if max iterations are reached

# Choose an index to apply Newton's method (e.g., for the 50th data point)
index_to_find = 50

# Set initial guess for Newton's method
x0 = 50

# Measure computational speed (time)
start_time = time.time()

# Find the root using Newton's method
root, iterations = newtons_method(residual_function, residual_function_derivative, x0, tol=1e-6, index=index_to_find)

# End computational time
end_time = time.time()
computational_time = end_time - start_time

# Compute accuracy by evaluating the residual at the root
accuracy = abs(residual_function(root, index_to_find))

# Plot actual data and exponential model
plt.figure(figsize=(10, 6))
plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)
plt.plot(X_data, A * np.exp(B * X_data), color='red', label=f'Exponential Fit: y = {A:.4f} * exp({B:.4f} * x)', linewidth=2)

plt.xlabel('Minute')
plt.ylabel('Output (kW)')
plt.title("Exponential Regression with Newton's Method Root Calculation")
plt.legend()
plt.show()

# Print the results
print(f"Root for data point {index_to_find} found at x = {root}")
print(f"Convergence: Number of iterations = {iterations}")
print(f"Accuracy (Residual at the root): {accuracy}")
print(f"Computational Speed: {computational_time:.6f} seconds")





##### Task 3: Exponential Model #####
  ##### Newton-Raphson Method #####

import numpy as np
import matplotlib.pyplot as plt
import time  # To measure computational speed

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

Y_data = np.array([
    0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
    7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
    9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
    16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
    25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
    37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
    55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
    71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
    95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
    113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
    141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
    172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
    200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
    233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
    265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
    304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
    343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
    388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
    434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
    479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747
])

# Logarithmic transformation of Y_data to fit the exponential model
Y_log = np.log(Y_data)

# Perform linear regression on the log-transformed data
coeffs = np.polyfit(X_data, Y_log, 1)

# Exponential model: y = A * exp(B * x)
A = np.exp(coeffs[1])  # exp(intercept)
B = coeffs[0]  # slope

# Predict Y values using the exponential model
Y_exp_pred = A * np.exp(B * X_data)

# Define residual function (difference between actual and predicted values)
def residual_function(x, index):
    return Y_data[index] - (A * np.exp(B * x))

# Derivative of the residual function
def residual_function_derivative(x, index):
    return -A * B * np.exp(B * x)  # Derivative of A * exp(B * x)

# Newton-Raphson Method implementation
def newton_raphson_method(func, func_prime, x0, tol=1e-6, max_iterations=100, index=None):
    """
    Newton-Raphson method to find the root of a function func with its derivative func_prime.
    Tracks the number of iterations and convergence.
    """
    iteration = 0
    x = x0
    
    while iteration < max_iterations:
        fx = func(x, index)
        f_prime_x = func_prime(x, index)
        
        # Avoid division by zero
        if abs(f_prime_x) < 1e-12:
            raise ValueError("Derivative near zero; Newton-Raphson method failed.")
        
        # Newton-Raphson update
        x_new = x - fx / f_prime_x
        
        # Check convergence
        if abs(x_new - x) < tol:
            return x_new, iteration  # Return the root and number of iterations
        
        x = x_new
        iteration += 1
    
    return x, iteration  # Return root and number of iterations if max iterations are reached

# Choose an index to apply the Newton-Raphson method (e.g., for the 50th data point)
index_to_find = 50

# Initial guess for Newton-Raphson
initial_guess = 50

# Measure computational speed (time)
start_time = time.time()

# Find the root using Newton-Raphson method
root, iterations = newton_raphson_method(residual_function, residual_function_derivative, initial_guess, tol=1e-6, index=index_to_find)

# End computational time
end_time = time.time()
computational_time = end_time - start_time

# Compute accuracy by evaluating the residual at the root
accuracy = abs(residual_function(root, index_to_find))

# Plot actual data and exponential model
plt.figure(figsize=(10, 6))
plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)
plt.plot(X_data, A * np.exp(B * X_data), color='red', label=f'Exponential Fit: y = {A:.4f} * exp({B:.4f} * x)', linewidth=2)

plt.xlabel('Minute')
plt.ylabel('Output (kW)')
plt.title('Exponential Regression with Newton-Raphson Method Root Calculation')
plt.legend()
plt.show()

# Print the results
print(f"Root for data point {index_to_find} found at x = {root}")
print(f"Convergence: Number of iterations = {iterations}")
print(f"Accuracy (Residual at the root): {accuracy}")
print(f"Computational Speed: {computational_time:.6f} seconds")





##### Task 3: Exponential Model #####
      ##### Secant Method #####
      
import numpy as np
import matplotlib.pyplot as plt
import time  # To measure computational speed

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

Y_data = np.array([
    0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
    7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
    9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
    16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
    25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
    37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
    55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
    71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
    95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
    113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
    141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
    172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
    200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
    233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
    265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
    304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
    343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
    388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
    434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
    479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747
])

# Logarithmic transformation of Y_data to fit the exponential model
Y_log = np.log(Y_data)

# Perform linear regression on the log-transformed data
coeffs = np.polyfit(X_data, Y_log, 1)

# Exponential model: y = A * exp(B * x)
A = np.exp(coeffs[1])  # exp(intercept)
B = coeffs[0]  # slope

# Predict Y values using the exponential model
Y_exp_pred = A * np.exp(B * X_data)

# Define residual function (difference between actual and predicted values)
def residual_function(x, index):
    return Y_data[index] - (A * np.exp(B * x))

# Secant Method implementation
def secant_method(func, x0, x1, tol=1e-6, max_iterations=100, index=None):
    """
    Secant method to find the root of a function func.
    It does not require the derivative, only function evaluations.
    """
    iteration = 0
    while iteration < max_iterations:
        fx0 = func(x0, index)
        fx1 = func(x1, index)
        
        # Secant update
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        
        # Check convergence
        if abs(x_new - x1) < tol:
            return x_new, iteration  # Return the root and number of iterations
        
        x0, x1 = x1, x_new
        iteration += 1
    
    return x1, iteration  # Return root and number of iterations if max iterations are reached

# Choose an index to apply the Secant method (e.g., for the 50th data point)
index_to_find = 50

# Initial guesses for Secant method
x0 = 45  # First guess
x1 = 50  # Second guess

# Measure computational speed (time)
start_time = time.time()

# Find the root using Secant method
root, iterations = secant_method(residual_function, x0, x1, tol=1e-6, index=index_to_find)

# End computational time
end_time = time.time()
computational_time = end_time - start_time

# Compute accuracy by evaluating the residual at the root
accuracy = abs(residual_function(root, index_to_find))

# Plot actual data and exponential model
plt.figure(figsize=(10, 6))
plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)
plt.plot(X_data, A * np.exp(B * X_data), color='red', label=f'Exponential Fit: y = {A:.4f} * exp({B:.4f} * x)', linewidth=2)

plt.xlabel('Minute')
plt.ylabel('Output (kW)')
plt.title('Exponential Regression with Secant Method Root Calculation')
plt.legend()
plt.show()

# Print the results
print(f"Root for data point {index_to_find} found at x = {root}")
print(f"Convergence: Number of iterations = {iterations}")
print(f"Accuracy (Residual at the root): {accuracy}")
print(f"Computational Speed: {computational_time:.6f} seconds")





##### Task 5: Interpolation Between Data Point #####
       ##### Cubic Spline Interpolation #####

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Original Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

Y_data = np.array([0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
                   7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
                   9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
                   16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
                   25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
                   37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
                   55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
                   71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
                   95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
                   113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
                   141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
                   172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
                   200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
                   233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
                   265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
                   304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
                   343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
                   388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
                   434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
                   479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747])

# Create the cubic spline interpolation function
cubic_spline = CubicSpline(X_data, Y_data)

# Generate new X values at 0.5 intervals between 0 and 99
X_new = np.arange(X_data.min(), X_data.max(), 0.5)

# Interpolated Y values at new X values using cubic spline
Y_new = cubic_spline(X_new)

# Plot original and interpolated data
plt.figure(figsize=(10, 6))
plt.plot(X_data, Y_data, 'o', label='Original Data', color='yellow')
plt.plot(X_new, Y_new, '-', label='Cubic Spline Interpolation', color='red')

# Highlight the interpolated value at minute 7.5
plt.scatter(7.5, cubic_spline(7.5), color='blue', label='Interpolated Value at Minute 7.5', s=200)

plt.xlabel('Minute')
plt.ylabel('PV Power Output (kW)')
plt.title('Cubic Spline Interpolation for PV Power Output')
plt.legend()
plt.show()

# Print interpolated value at minute 7.5
print(f"Interpolated value at minute 7.5: {cubic_spline(7.5)}")






##### Task 5: Interpolation Between Data Point #####
          ##### Linear Interpolation #####

import numpy as np
import matplotlib.pyplot as plt

# Original Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

Y_data = np.array([0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
                   7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
                   9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
                   16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
                   25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
                   37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
                   55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
                   71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
                   95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
                   113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
                   141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
                   172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
                   200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
                   233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
                   265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
                   304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
                   343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
                   388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
                   434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
                   479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747])

# Linear Interpolation function
def linear_interpolation(x, X_data, Y_data):
    # Find the interval in which x lies
    for i in range(len(X_data) - 1):
        if X_data[i] <= x <= X_data[i + 1]:
            # Perform linear interpolation
            y = Y_data[i] + (Y_data[i + 1] - Y_data[i]) * (x - X_data[i]) / (X_data[i + 1] - X_data[i])
            return y
    return None  # Return None if x is outside the bounds

# Generate new X values at 0.5 intervals between 0 and 99
X_new = np.arange(X_data.min(), X_data.max(), 0.5)

# Interpolated Y values at new X values using linear interpolation
Y_new = [linear_interpolation(x, X_data, Y_data) for x in X_new]

# Plot original and interpolated data
plt.figure(figsize=(10, 6))
plt.plot(X_data, Y_data, 'o', label='Original Data', color='yellow')
plt.plot(X_new, Y_new, '-', label='Linear Interpolation', color='green')

# Highlight the interpolated value at minute 7.5
plt.scatter(7.5, linear_interpolation(7.5, X_data, Y_data), color='blue', label='Interpolated Value at Minute 7.5', s=200)

plt.xlabel('Minute')
plt.ylabel('PV Power Output (kW)')
plt.title('Linear Interpolation for PV Power Output')
plt.legend()
plt.show()

# Print interpolated value at minute 7.5
print(f"Interpolated value at minute 7.5: {linear_interpolation(7.5, X_data, Y_data)}")





##### Task 5: Interpolation Between Data Point #####
        ##### Quadratic Interpolation #####

import numpy as np
import matplotlib.pyplot as plt

# Original Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

Y_data = np.array([0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
                   7.722371802, 5.149155626, 6.265100781, 6.651108168, 4.112457569, 
                   9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
                   16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
                   25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
                   37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
                   55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
                   71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
                   95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
                   113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
                   141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
                   172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
                   200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
                   233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
                   265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
                   304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
                   343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
                   388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
                   434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
                   479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747])

# Quadratic Interpolation function
def quadratic_interpolation(x, X_data, Y_data):
    # Find the interval in which x lies, use 3 points for quadratic interpolation
    for i in range(len(X_data) - 2):
        if X_data[i] <= x <= X_data[i + 2]:
            # Get the three points around x for the quadratic interpolation
            x0, x1, x2 = X_data[i], X_data[i + 1], X_data[i + 2]
            y0, y1, y2 = Y_data[i], Y_data[i + 1], Y_data[i + 2]
            
            # Quadratic interpolation formula
            L0 = ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2))
            L1 = ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2))
            L2 = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1))
            
            # Interpolated value
            y = L0 * y0 + L1 * y1 + L2 * y2
            return y
    return None  # Return None if x is outside the bounds

# Generate new X values at 0.5 intervals between 0 and 99
X_new = np.arange(X_data.min(), X_data.max(), 0.5)

# Interpolated Y values at new X values using quadratic interpolation
Y_new = [quadratic_interpolation(x, X_data, Y_data) for x in X_new]

# Plot original and interpolated data
plt.figure(figsize=(10, 6))
plt.plot(X_data, Y_data, 'o', label='Original Data', color='yellow')
plt.plot(X_new, Y_new, '-', label='Quadratic Interpolation', color='purple')

# Highlight the interpolated value at minute 7.5
plt.scatter(7.5, quadratic_interpolation(7.5, X_data, Y_data), color='blue', label='Interpolated Value at Minute 7.5', s=200)

plt.xlabel('Minute')
plt.ylabel('PV Power Output (kW)')
plt.title('Quadratic Interpolation for PV Power Output')
plt.legend()
plt.show()

# Print interpolated value at minute 7.5
print(f"Interpolated value at minute 7.5: {quadratic_interpolation(7.5, X_data, Y_data)}")





##### Task 6: Best-fit regression Using Interpolated Data #####
  ##### Quadratic Interpolation using Polynomial Method #####

import numpy as np
import matplotlib.pyplot as plt

# Data
X_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 7.5, 8, 9, 
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

Y_data = np.array([
    0.358709354, 0.864570967, 0.395445462, 2.227428577, 4.008101714,
    7.722371802, 5.149155626, 6.265100781, 7.0578555788101465, 6.651108168, 4.112457569, 
    9.146972249, 10.67046042, 16.92648422, 13.16527807, 15.80309468, 
    16.88057646, 16.46264392, 23.03564563, 24.30386607, 26.53206389,
    25.38122509, 32.35558862, 29.19629787, 35.72371419, 41.58091125,
    37.96892735, 41.66740454, 45.94930273, 47.79304869, 48.84867314,
    55.33712595, 56.42539257, 62.94718486, 63.71115153, 72.29986881,
    71.38349342, 76.15587697, 82.37703443, 82.33827137, 89.40491987,
    95.81428551, 94.33503353, 102.3692677, 107.0697656, 112.7636457, 
    113.4760986, 118.1590868, 126.7938831, 131.3939693, 136.4509857,
    141.8928964, 145.1899506, 152.4645074, 158.1361449, 161.7712972,
    172.681549, 175.7476658, 178.367393, 188.1131072, 191.0006367,
    200.7741692, 207.8671912, 210.3586353, 220.4767523, 226.0255619,
    233.5941203, 242.593586, 245.2592238, 251.2925277, 258.1709711, 
    265.5683794, 274.3957966, 282.6823039, 290.1033816, 298.8543665,
    304.9760038, 315.7070682, 320.2206863, 334.2403383, 338.2013347,
    343.4856849, 351.408215, 362.9649448, 370.1030744, 380.628001,
    388.8964752, 396.6543422, 404.0564126, 411.7703056, 423.0569701, 
    434.9127976, 442.9781875, 449.5085224, 461.8963619, 471.9706348,
    479.1822851, 491.1074502, 500.8664174, 508.5140594, 521.6655747
])

# Function to compute R-squared
def compute_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Function to compute Standard Error
def compute_standard_error(y_true, y_pred, num_params):
    n = len(y_true)  # Number of data points
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    standard_error = np.sqrt(residual_sum_of_squares / (n - num_params))
    return standard_error

# Fit polynomial models and calculate R-squared and Standard Error
orders = [2, 3, 4, 5]
r_squared_values = []
standard_errors = []

for order in orders:
    coeffs = np.polyfit(X_data, Y_data, order)
    Y_poly_pred = np.polyval(coeffs, X_data)
    r_squared = compute_r_squared(Y_data, Y_poly_pred)
    r_squared_values.append(r_squared)
    
    # Compute standard error
    standard_error = compute_standard_error(Y_data, Y_poly_pred, order + 1)  # num_params = order + 1 (including intercept)
    standard_errors.append(standard_error)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(X_data, Y_data, color='blue', label='Actual Data', alpha=0.7)
    plt.plot(X_data, Y_poly_pred, label=f'Polynomial Order {order}', color='pink', linewidth=1.5)
    plt.xlabel('Minute')
    plt.ylabel('Output (kW)')
    plt.title(f'Polynomial Regression (Order {order})')
    plt.legend()
    plt.show()

    print(f"Order {order}: Coefficients = {coeffs}")
    print(f"Order {order}: R-squared = {r_squared}")
    print(f"Order {order}: Standard Error = {standard_error}")

# Print the R-squared values and Standard Errors
for order, r_squared, std_err in zip(orders, r_squared_values, standard_errors):
    print(f"Polynomial Order {order}: R-squared = {r_squared}, Standard Error = {std_err}")
