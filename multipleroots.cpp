#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;

const double q = 1.6022e-19;  // Charge of an electron (C)
const double kB = 1.3906e-23; // Boltzmann's constant (J/K)
const double T = 297;         // Temperature (K)
const double VOC = 0.5;       // Open-circuit voltage (V)
const double epsilon = 1e-6;  // Convergence threshold

// Function to compute f(Vmp)
double f(double Vmp) {
    double exp_term = exp(q * Vmp / (kB * T));
    return exp_term * (1 + (q * Vmp) / (kB * T)) - exp(q * VOC / (kB * T));
}

// Derivative of the function
double df(double Vmp) {
    double exp_term = exp(q * Vmp / (kB * T));
    return (q / (kB * T)) * (exp_term * (1 + (q * Vmp) / (kB * T)) + exp_term);
}

int main() {
    double Vmp = 0.5; // Initial guess
    double Vmp_prev;
    double error;
    int iteration = 0;

    cout << fixed << setprecision(8);
    
    cout << "| Iteration |      Vmp       |   Error (%)   |\n";
    cout << "---------------------------------------------\n";

    do {
        Vmp_prev = Vmp;
        double f_val = f(Vmp);
        double df_val = df(Vmp);

        if (df_val == 0) {
            cerr << "Derivative is zero. Newton-Raphson method fails." << endl;
            return 1;
        }

        Vmp = Vmp_prev - f_val / df_val;

        error = fabs((Vmp - Vmp_prev) / Vmp_prev) * 100.0;

        cout << "| " << setw(10) << iteration
             << " | " << setw(12) << Vmp
             << " | " << setw(12) << error << " |\n";

        iteration++;

    } while (error > epsilon);

    cout << "Converged Vmp: " << Vmp << endl;
    cout << "Final Error (%): " << error << endl;

    return 0;
}
