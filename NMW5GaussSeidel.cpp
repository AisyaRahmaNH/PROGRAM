#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;

// Fungsi untuk menghitung error relatif
double calculateError(double oldValue, double newValue) {
    return fabs((newValue - oldValue) / newValue) * 100;
}

int main() {
    // Deklarasi variabel
    double c1 = 0, c2 = 0, c3 = 0;  // Nilai awal (iterasi ke-0)
    double c1_old, c2_old, c3_old;  // Menyimpan nilai sebelumnya untuk menghitung error
    double err_c1, err_c2, err_c3;  // Error untuk masing-masing variabel
    int maxIterations = 10;          // Maksimal iterasi adalah 10 kali
    int iteration = 0;               // Variabel untuk melacak iterasi

    cout << fixed << setprecision(6);

    // Loop iterasi Gauss-Seidel
    while (iteration < maxIterations) {
        iteration++;

        // Menyimpan nilai sebelumnya
        c1_old = c1;
        c2_old = c2;
        c3_old = c3;

        // Update nilai variabel menggunakan metode Gauss-Seidel
        c1 = (1.0 / 15.0) * (3800 + 3 * c2 + c3);    // Persamaan untuk c1
        c2 = (1.0 / 18.0) * (1200 + 3 * c1 + 6 * c3);  // Persamaan untuk c2
        c3 = (1.0 / 12.0) * (2350 + 4 * c1 + c2);    // Persamaan untuk c3

        // Menghitung error relatif setelah iterasi pertama
        if (iteration > 1) {
            err_c1 = calculateError(c1_old, c1);
            err_c2 = calculateError(c2_old, c2);
            err_c3 = calculateError(c3_old, c3);
        } else {
            err_c1 = err_c2 = err_c3 = 0;  // Error iterasi pertama selalu 0
        }

        // Output hasil dari iterasi 1 sampai 9
        if (iteration < 10) {
            cout << "Iteration " << iteration << ":" << endl;
            cout << "c1 = " << c1 << " (Error: " << err_c1 << "%)" << endl;
            cout << "c2 = " << c2 << " (Error: " << err_c2 << "%)" << endl;
            cout << "c3 = " << c3 << " (Error: " << err_c3 << "%)" << endl;
            cout << "--------------------------------------------" << endl;
        }
    }

    // Output hasil akhir setelah 10 iterasi
    cout << "Solusi akhir setelah 9 iterasi :" << endl;
    cout << "c1 = " << c1 << endl;
    cout << "c2 = " << c2 << endl;
    cout << "c3 = " << c3 << endl;

    return 0;
}
