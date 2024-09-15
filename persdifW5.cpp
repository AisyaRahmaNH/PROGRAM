#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// Function to solve the differential equation for distinct real roots
double solveDistinctRoots(double x, double C1, double C2) {
    double exp3x = exp(3 * x);
    double exp2x = exp(2 * x);
    return C1 * exp3x + C2 * exp2x;
}

// Function to solve the differential equation for repeated roots
double solveRepeatedRoots(double x, double C1, double C2) {
    double exp2x = exp(2 * x);
    return (C1 + C2 * x) * exp2x;
}

// Function to solve the differential equation for complex roots
double solveComplexRoots(double x, double C1, double C2) {
    double expNegX = exp(-x);
    double cos2x = cos(2 * x);
    double sin2x = sin(2 * x);
    return expNegX * (C1 * cos2x + C2 * sin2x);
}

int main() {
    // Parameters for each differential equation
    double C1_distinct = 2, C2_distinct = -1; // For distinct real roots
    double C1_repeated = 2, C2_repeated = -3; // For repeated roots
    double C1_complex = 1, C2_complex = 0.5;  // For complex roots

    // Define the range of x values for testing
    vector<double> x_values = {0, 1, 2, 3, 4};

    // Display the results for the distinct real roots equation
    cout << "Distinct Real Roots Solution:" << endl;
    for (double x : x_values) {
        double result = solveDistinctRoots(x, C1_distinct, C2_distinct);
        cout << "y(" << x << ") = " << result << endl;
    }

    // Display the results for the repeated roots equation
    cout << "\nRepeated Roots Solution:" << endl;
    for (double x : x_values) {
        double result = solveRepeatedRoots(x, C1_repeated, C2_repeated);
        cout << "y(" << x << ") = " << result << endl;
    }

    // Display the results for the complex roots equation
    cout << "\nComplex Roots Solution:" << endl;
    for (double x : x_values) {
        double result = solveComplexRoots(x, C1_complex, C2_complex);
        cout << "y(" << x << ") = " << result << endl;
    }

    return 0;
}
