#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

// Fungsi untuk menghitung error relatif
double calculateError(double oldValue, double newValue) {
    return fabs((newValue - oldValue) / newValue) * 100;
}

// Fungsi untuk melakukan LU Decomposition
void luDecomposition(double A[3][3], double L[3][3], double U[3][3], int n) {
    for (int i = 0; i < n; i++) {
        // Upper Triangular
        for (int j = i; j < n; j++) {
            U[i][j] = A[i][j];
            for (int k = 0; k < i; k++) {
                U[i][j] -= L[i][k] * U[k][j];
            }
        }
        // Lower Triangular
        for (int j = i; j < n; j++) {
            if (i == j) {
                L[i][i] = 1; // Diagonal sebagai 1
            } else {
                L[j][i] = A[j][i];
                for (int k = 0; k < i; k++) {
                    L[j][i] -= L[j][k] * U[k][i];
                }
                L[j][i] /= U[i][i];
            }
        }
    }
}

// Fungsi untuk menyelesaikan sistem menggunakan Forward Substitution
void forwardSubstitution(double L[3][3], double b[3], double y[3]) {
    for (int i = 0; i < 3; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++) {
            y[i] -= L[i][j] * y[j];
        }
    }
}

// Fungsi untuk menyelesaikan sistem menggunakan Back Substitution
void backSubstitution(double U[3][3], double y[3], double x[3]) {
    for (int i = 2; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < 3; j++) {
            x[i] -= U[i][j] * x[j];
        }
        x[i] /= U[i][i];
    }
}

int main() {
    // Matriks koefisien A dan vektor b
    double A[3][3] = {
        {15, -3, -1},
        {-3, 18, -6},
        {-4, -1, 12}
    };
    double b[3] = {3800, 1200, 2350};

    // Matriks L dan U
    double L[3][3] = {0}, U[3][3] = {0};
    double x[3] = {0}, y[3] = {0};
    double x_old[3] = {0}; // Menyimpan nilai sebelumnya untuk menghitung error
    double err[3] = {0};
    double epsilon_s = 5.0; // Toleransi error relatif
    int maxIterations = 10;  // Maksimal iterasi
    int iteration = 0;       // Variabel untuk melacak iterasi

    // Melakukan LU Decomposition
    luDecomposition(A, L, U, 3);

    cout << fixed << setprecision(6);

    // Loop untuk iterasi
    while (iteration < maxIterations) {
        iteration++;

        // Selesaikan L * y = b
        forwardSubstitution(L, b, y);

        // Selesaikan U * x = y
        backSubstitution(U, y, x);

        // Menghitung error relatif
        for (int i = 0; i < 3; i++) {
            if (iteration > 1) {
                err[i] = calculateError(x_old[i], x[i]);
            } else {
                err[i] = 0; // Error iterasi pertama selalu 0
            }
            x_old[i] = x[i]; // Simpan nilai sebelumnya
        }

        // Output hasil dari iterasi 1 sampai 9
        if (iteration < 10) {
            cout << "Iteration " << iteration << ":" << endl;
            cout << "c1 = " << x[0] << " (Error: " << err[0] << "%)" << endl;
            cout << "c2 = " << x[1] << " (Error: " << err[1] << "%)" << endl;
            cout << "c3 = " << x[2] << " (Error: " << err[2] << "%)" << endl;
            cout << "--------------------------------------------" << endl;
        }
    }

    // Output hasil akhir setelah 9 iterasi
    cout << "Solusi akhir setelah 9 iterasi :" << endl;
    cout << "c1 = " << x[0] << endl;
    cout << "c2 = " << x[1] << endl;
    cout << "c3 = " << x[2] << endl;

    return 0;
}
