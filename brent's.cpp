#include <iostream>
#include <cmath>
#include <limits>
#include <iomanip>
using namespace std;

// Constants
const double q = 1.6022e-19; // Charge of electron in Coulombs
const double kB = 1.3906e-23; // Boltzmann constant in J/K
const double VOC = 0.5;       // Open circuit voltage in Volts
const double T = 297;         // Temperature in Kelvin
const double epsilon = 1e-6;  // Precision for termination

// Function to compute f(Vmp)
double f(double Vmp) {
    double expTerm = exp((q * Vmp) / (kB * T));
    double rightTerm = exp((q * VOC) / (kB * T));
    return expTerm * (1 + (q * Vmp) / (kB * T)) - rightTerm;
}

// Brent's method implementation
double brent(double a, double b) {
    const double tol = 1e-10; // Tolerance for convergence
    double fa = f(a);
    double fb = f(b);
    double c = a, fc = fa;
    bool mflag = true;
    double s, fs, d = 0.0, e = 0.0;

    if (fa * fb > 0.0) {
        cerr << "Error: f(a) and f(b) must have different signs" << endl;
        return numeric_limits<double>::quiet_NaN();
    }

    const int width = 15; // Width of each column for formatting
    cout << left << setw(5) << "Iter" 
         << "| " << setw(width) << "a" 
         << "| " << setw(width) << "b" 
         << "| " << setw(width) << "c" 
         << "| " << setw(width) << "Vmp" 
         << "| " << setw(width) << "f(Vmp)" 
         << "| " << setw(width) << "Error (%)" << endl;
    
    // Print the separator line
    cout << left << setw(5) << string(5, '-') 
         << "--" << setw(width) << string(width, '-') 
         << "--" << setw(width) << string(width, '-') 
         << "--" << setw(width) << string(width, '-') 
         << "--" << setw(width) << string(width, '-') 
         << "--" << setw(width) << string(width, '-') 
         << "--" << setw(width) << string(width, '-') << endl;

    int iter = 0;
    while (fabs(b - a) > tol) {
        iter++;
        double fa = f(a);
        double fb = f(b);
        double Vmp;
        
        if (fabs(fa - fb) < tol) {
            s = (a + b) / 2;
        } else {
            double q1 = (a * fb - b * fa) / (fb - fa);
            double q2 = (a - b) / (fb - fa);
            double q3 = (q1 - b) / (q2 + q2);
            s = b - (fb * q3) / (fb - fa);
        }

        if ((s < (3 * a + b) / 4) || (s > b)) s = (a + b) / 2;

        fs = f(s);
        if (fs == 0) return s;

        if (fa * fs < 0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if (fabs(b - a) < tol) {
            Vmp = (a + b) / 2;
        }

        double error = fabs((b - a) / (a + b)) * 100.0;
        cout << left << setw(5) << iter 
             << "| " << setw(width) << a 
             << "| " << setw(width) << b 
             << "| " << setw(width) << c 
             << "| " << setw(width) << Vmp 
             << "| " << setw(width) << fs 
             << "| " << setw(width) << error << endl;

        if (fabs(fs) < epsilon) break;
    }

    return (a + b) / 2;
}

int main() {
    double a = 0.4; // Initial bracket points
    double b = 0.6;
    double Vmp = brent(a, b);
    cout << "Vmp (voltage at maximum power) = " << Vmp << " V" << endl;
    return 0;
}
