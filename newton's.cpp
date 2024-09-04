#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;

// Constants
const double q = 1.6022e-19; // Charge of an electron in Coulombs
const double kB = 1.3906e-23; // Boltzmann's constant in J/K
const double T = 297;         // Temperature in Kelvin
const double VOC = 0.5;       // Open circuit voltage in Volts
const double epsilon = 1e-6;  // Convergence tolerance

// Function f(Vmp)
double f(double Vmp) {
    return exp(q * Vmp / (kB * T)) * (1 + q * Vmp / (kB * T)) - exp(q * VOC / (kB * T));
}

// Derivative f'(Vmp)
double f_prime(double Vmp) {
    double exp1 = exp(q * Vmp / (kB * T));
    double factor = q / (kB * T);
    return factor * exp1 * (1 + q * Vmp / (kB * T)) + exp1 * factor;
}

// Newton's Method
int main() {
    double Vmp = 0.5;       // Initial guess
    double Vmp_old;
    int iteration = 0;

    cout << fixed << setprecision(6);
    cout << "| Iteration |    Vmp     |       f(Vmp)     |      f'(Vmp)      |    Error (%)    |" << endl;
    cout << "-----------------------------------------------------------------------------------" << endl;
    
    do {
        Vmp_old = Vmp;
        double fVmp = f(Vmp_old);
        double fVmp_prime = f_prime(Vmp_old);

        // Avoid division by zero
        if (fabs(fVmp_prime) < 1e-12) {
            cerr << "Derivative too small, stopping iteration.\n";
            break;
        }

        // Update Vmp using Newton-Raphson formula
        Vmp = Vmp_old - fVmp / fVmp_prime;

        // Calculate relative error
        double error = fabs((Vmp - Vmp_old) / Vmp_old) * 100;

        // Print current iteration details
        cout << "| " << setw(9) << iteration++
             << " | " << setw(10) << Vmp
             << " | " << setw(16) << fVmp
             << " | " << setw(16) << fVmp_prime
             << " | " << setw(13) << error << " |" << endl;
        
    } while (fabs((Vmp - Vmp_old) / Vmp_old) > epsilon);
    
    cout << "\nConverged Vmp = " << Vmp << " V after " << iteration << " iterations.\n";
    return 0;
}
