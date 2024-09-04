#include <iostream>
#include <cmath>
#include <iomanip>
using namespace std;

// Constants
const double q = 1.6022e-19; // Charge of electron (Coulombs)
const double kB = 1.3906e-23; // Boltzmann's constant (J/K)
const double VOC = 0.5; // Open circuit voltage (V)
const double T = 297.0; // Temperature (K)
const double epsilon = 1e-6; // Convergence tolerance

// Function to compute f(Vmp)
double f(double Vmp) {
    double exponent = q * Vmp / (kB * T);
    return exp(exponent) * (1 + exponent) - exp(q * VOC / (kB * T));
}

int main() {
    double Vmp1 = 0.5; // Initial guess 1
    double Vmp0 = 0.4; // Initial guess 2 (different from Vmp1)
    double fVmp1 = f(Vmp1);
    double fVmp0 = f(Vmp0);
    double Vmp_next;
    int iteration = 0;
    
    cout << fixed << setprecision(8);
    cout << "| Iteration  |    Vmp     |           f(Vmp)           |   Error (%)     |\n";
    cout << "--------------------------------------------------------------------------\n";
    
    while (fabs((Vmp1 - Vmp0) / Vmp1) > epsilon) {
        // Compute the next Vmp using the secant method formula
        Vmp_next = Vmp1 - fVmp1 * (Vmp1 - Vmp0) / (fVmp1 - fVmp0);
        double fVmp_next = f(Vmp_next);
        
        // Compute the relative error
        double error = fabs((Vmp_next - Vmp1) / Vmp_next) * 100;
        
        // Print the current iteration details
        cout << "| " << setw(10) << iteration
                  << " | " << setw(10) << Vmp1
                  << " | " << setw(26) << fVmp1
                  << " | " << setw(15) << error << " |\n";
        
        // Update values for the next iteration
        Vmp0 = Vmp1;
        Vmp1 = Vmp_next;
        fVmp0 = fVmp1;
        fVmp1 = fVmp_next;
        
        iteration++;
    }
    
    // Print final result
    cout << "Converged Vmp: " << Vmp1 << "\n";
    cout << "Final Error (%): " << fabs((Vmp1 - Vmp0) / Vmp1) * 100 << "\n";
    
    return 0;
}
