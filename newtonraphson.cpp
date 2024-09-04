#include <iostream>
#include <cmath>
#include <iomanip>
using namespace std;

// Constants
const double q = 1.6022e-19; // Charge on an electron in Coulombs
const double kB = 1.3906e-23; // Boltzmann's constant in J/K
const double VOC = 0.5; // Open circuit voltage in Volts
const double T = 297; // Temperature in Kelvin
const double epsilon = 1e-6; // Convergence criterion

// Function f(V)
double f(double V) {
    double exp1 = exp(q * V / (kB * T));
    double exp2 = exp(q * VOC / (kB * T));
    return exp1 * (1 + q * V / (kB * T)) - exp2;
}

// Derivative f'(V)
double f_prime(double V) {
    double exp1 = exp(q * V / (kB * T));
    return exp1 * (q / (kB * T) * (1 + q * V / (kB * T)) + q / (kB * T));
}

int main() {
    double V = 0.5; // Initial guess
    double V_next;
    int iteration = 0;

    cout << std::fixed << setprecision(6);
    cout << "Iteration |     Vmp    |       f(Vmp)     |       f'(Vmp)      | Error (%)" << endl;
    cout << "---------------------------------------------------------------------------" << endl;

    while (true) {
        double f_V = f(V);
        double f_prime_V = f_prime(V);

        // Avoid division by zero
        if (f_prime_V == 0) {
            std::cerr << "Derivative is zero. Newton-Raphson method fails." << endl;
            return 1;
        }

        V_next = V - f_V / f_prime_V;
        double error = std::abs((V_next - V) / V) * 100;

        // Print the iteration details with column delimiters
        cout << setw(9) << iteration + 1
                  << " | " << setw(10) << V
                  << " | " << setw(10) << f_V
                  << " | " << setw(10) << f_prime_V
                  << " | " << setw(9) << error << "%" << endl;

        // Check for convergence
        if (error <= epsilon) {
            break;
        }

        // Update V for the next iteration
        V = V_next;
        iteration++;
    }

    cout << "Converged to Vmp = " << V << " Volts" << endl;
    return 0;
}
