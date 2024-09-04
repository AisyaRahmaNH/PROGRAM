#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;

// Constants
const double q = 1.6022e-19; // Charge of an electron in Coulombs
const double kB = 1.3906e-23; // Boltzmann constant in J/K
const double VOC = 0.5;       // Open circuit voltage in Volts
const double T = 297;         // Temperature in Kelvin
const double epsilon = 1e-6;  // Convergence criterion

// Function to compute the value of f(Vmp)
double f(double Vmp) {
    double exp1 = exp(q * Vmp / (kB * T));
    double exp2 = exp(q * VOC / (kB * T));
    return exp1 * (1 + (q * Vmp) / (kB * T)) - exp2;
}

// Bisection method
double bisectionMethod(double Vlow, double Vhigh) {
    double Vmp, f_low, f_mid, f_high;
    int iteration = 0;
    
    // Table header with borders
    cout << "| Iteration   |      Vlow      |      Vhigh     |      Vmid      |       f(Vmid)        |" << endl;
    cout << "------------------------------------------------------------------------------------" << endl;
    
    do {
        Vmp = (Vlow + Vhigh) / 2.0;
        f_low = f(Vlow);
        f_mid = f(Vmp);
        f_high = f(Vhigh);
        
        double prev_Vmp = Vmp;
        
        if (f_low * f_mid < 0) {
            Vhigh = Vmp;
        } else {
            Vlow = Vmp;
        }
        
        double error = fabs((Vmp - prev_Vmp) / Vmp) * 100;
        
        // Print the current iteration results with borders
        cout << "| " << setw(11) << iteration++
             << " | " << setw(14) << Vlow
             << " | " << setw(14) << Vhigh
             << " | " << setw(14) << Vmp
             << " | " << setw(20) << f_mid
             << " |" << endl;
        
        
        if (fabs(f_mid) < epsilon) {
            break;
        }
        
    } while (fabs((Vhigh - Vlow) / Vmp) > epsilon);
    
    return Vmp;
}

int main() {
    double Vlow = 0.0;
    double Vhigh = 0.5; // Initial guess range
    
    double Vmp = bisectionMethod(Vlow, Vhigh);
    
    cout << "The voltage Vmp at which the power output is maximized is: " << Vmp << " V\n";
    
    return 0;
}
