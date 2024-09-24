import numpy as np                                                                  # Import numpy untuk melakukan operasi numerik
import matplotlib.pyplot as plt                                                     # Import matplotlib untuk membuat plot
from scipy.optimize import curve_fit                                                # Import curve_fit dari scipy untuk melakukan fitting data ke model regresi

# Data untuk x (hari) dan y (jumlah sel dalam jutaan)
x = np.array([0, 4, 8, 12, 16, 20])                                                 # Waktu dalam hari
y = np.array([67, 84, 98, 125, 149, 185])                                           # Jumlah sel dalam jutaan

# Fungsi untuk model regresi linear: y = a + bx
def linear_regression(x, a, b):
    return a + b * x

# Fungsi untuk model regresi kuadratik: y = a + bx + cx^2
def quadratic_regression(x, a, b, c):
    return a + b * x + c * x**2

# Fungsi untuk model regresi eksponensial: y = a * exp(bx)
def exponential_regression(x, a, b):
    return a * np.exp(b * x)

# Melakukan fitting regresi linear pada data
params_linear, _ = curve_fit(linear_regression, x, y)                               # Memperoleh parameter a dan b
a_linear, b_linear = params_linear                                                  # Menyimpan hasil parameter a dan b untuk regresi linear

# Melakukan fitting regresi kuadratik pada data
params_quadratic, _ = curve_fit(quadratic_regression, x, y)                         # Memperoleh parameter a, b, dan c
a_quadratic, b_quadratic, c_quadratic = params_quadratic                            # Menyimpan hasil parameter a, b, dan c untuk regresi kuadratik

# Melakukan fitting regresi eksponensial pada data
params_exponential, _ = curve_fit(exponential_regression, x, y)                     # Memperoleh parameter a dan b
a_exponential, b_exponential = params_exponential                                   # Menyimpan hasil parameter a dan b untuk regresi eksponensial

# Membuat data x baru untuk prediksi garis regresi (nilai x lebih halus dari 0 sampai 40)
x_fit = np.linspace(0, 40, 400)                                                     # 400 titik data dari 0 hingga 40 hari untuk menggambar garis regresi yang halus

# Menghitung prediksi y untuk masing-masing model regresi dengan menggunakan x_fit
y_linear = linear_regression(x_fit, a_linear, b_linear)                             # Prediksi untuk regresi linear
y_quadratic = quadratic_regression(x_fit, a_quadratic, b_quadratic, c_quadratic)    # Prediksi untuk regresi kuadratik
y_exponential = exponential_regression(x_fit, a_exponential, b_exponential)         # Prediksi untuk regresi eksponensial

# Membuat plot
plt.figure(figsize=(10, 6))                                                         # Menentukan ukuran figure untuk plot

# Plot data asli (node data)
plt.scatter(x, y, color='red', label='Data Points')                                 # Menampilkan data asli sebagai titik merah

# Plot garis regresi linear
plt.plot(x_fit, y_linear, color='blue', label=f'Linear Regression: 
         y = {a_linear:.2f} + {b_linear:.2f}x')                                     # Garis biru untuk regresi linear

# Plot garis regresi kuadratik
plt.plot(x_fit, y_quadratic, color='green', label=f'Quadratic Regression: 
         y = {a_quadratic:.2f} + {b_quadratic:.2f}x + {c_quadratic:.2f}x^2')        # Garis hijau untuk regresi kuadratik

# Plot garis regresi eksponensial
plt.plot(x_fit, y_exponential, color='orange', label=f'Exponential Regression: 
         y = {a_exponential:.2f} * exp({b_exponential:.2f}x)')                      # Garis oranye untuk regresi eksponensial

# Menambahkan judul, label pada sumbu, dan legend
plt.title('Linear, Quadratic, and Exponential Regression')                          # Menambahkan judul pada plot
plt.xlabel('Days')                                                                  # Label untuk sumbu x (Hari)
plt.ylabel('Cell Count (in millions)')                                              # Label untuk sumbu y (Jumlah sel dalam jutaan)
plt.legend()                                                                        # Menampilkan legenda untuk tiap model regresi

# Menambahkan grid pada plot untuk mempermudah pembacaan
plt.grid(True)

# Menampilkan plot
plt.show()                                                                          # Memunculkan plot di layar
