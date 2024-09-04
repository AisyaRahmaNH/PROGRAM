#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;

// Constants
const double q = 1.6022e-19; // Charge of an electron (Coulombs)
const double kB = 1.3906e-23; // Boltzmann constant (J/K)
const double VOC = 0.5;       // Open circuit voltage (Volts)
const double T = 297;         // Temperature (Kelvin)
const double epsilon = 1e-6;  // Convergence criterion

int main() {
    // Initial guess for Vmp
    double Vmp = 0.5;
    double x_i = (q * Vmp) / (kB * T);
    double x_i1, Vmp1;
    double error;
    int iteration = 0;

    // Display table header
    cout << fixed << setprecision(6);
    cout << "Iteration |    Vmp   |    x_i    |   Error (%)" << endl;
    cout << "-----------------------------------------------------" << endl;

    do {
        iteration++;
        // Update x_i using the fixed-point iteration function
        x_i1 = (q * VOC) / (kB * T) - log(1 + x_i);
        Vmp1 = (kB * T * x_i1) / q;

        // Calculate error
        error = fabs((Vmp1 - Vmp) / Vmp) * 100;

        // Display the current iteration results
        cout << setw(9) << iteration << " | "
             << setw(8) << Vmp1 << " | "
             << setw(8) << x_i1 << " | "
             << setw(10) << error << "%" << endl;

        // Update for next iteration
        Vmp = Vmp1;
        x_i = x_i1;

    } while (error > epsilon);

    // Final result
    cout << "Converged Vmp = " << Vmp1 << " V" << endl;

    return 0;
}
