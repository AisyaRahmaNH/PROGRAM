#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// Function to compute quadratic regression coefficients
void quadraticRegression(const vector<double>& x, const vector<double>& y, double& a0, double& a1, double& a2) {
    double n = x.size();
    double Sx = 0, Sy = 0, Sxx = 0, Sxy = 0, Sxxx = 0, Sxxxx = 0, Sxxy = 0;
    
    for (int i = 0; i < n; ++i) {
        Sx += x[i];
        Sy += y[i];
        Sxx += x[i] * x[i];
        Sxxx += x[i] * x[i] * x[i];
        Sxxxx += x[i] * x[i] * x[i] * x[i];
        Sxy += x[i] * y[i];
        Sxxy += x[i] * x[i] * y[i];
    }

    double det = n * (Sxx * Sxxxx - Sxxx * Sxxx) - Sx * (Sx * Sxxxx - Sxx * Sxxx) + Sxx * (Sx * Sxxx - Sxx * Sxx);
    
    a0 = (Sy * (Sxx * Sxxxx - Sxxx * Sxxx) - Sx * (Sxy * Sxxxx - Sxxy * Sxxx) + Sxx * (Sxy * Sxxx - Sxxy * Sxx)) / det;
    a1 = (n * (Sxy * Sxxxx - Sxxy * Sxxx) - Sy * (Sx * Sxxxx - Sxx * Sxxx) + Sxx * (Sx * Sxxy - Sxx * Sxy)) / det;
    a2 = (n * (Sxx * Sxxy - Sxxx * Sxy) - Sx * (Sx * Sxxy - Sxx * Sxy) + Sy * (Sx * Sxxx - Sxx * Sxx)) / det;
}

// Function to calculate the predicted value using quadratic regression
double predict(double a0, double a1, double a2, double x) {
    return a0 + a1 * x + a2 * x * x;
}

// Function to compute statistical metrics
void calculateMetrics(const vector<double>& x, const vector<double>& y, double a0, double a1, double a2) {
    int n = x.size();
    double Sy = 0, Syx = 0, mean_y = 0, sum_error = 0, SStot = 0, SSres = 0, r_squared, r, Sy_estimate;

    for (int i = 0; i < n; ++i) {
        mean_y += y[i];
    }
    mean_y /= n;

    for (int i = 0; i < n; ++i) {
        double y_pred = predict(a0, a1, a2, x[i]);
        SStot += pow(y[i] - mean_y, 2);
        SSres += pow(y[i] - y_pred, 2);
        Sy += pow(y[i] - mean_y, 2);
        Syx += pow(y[i] - y_pred, 2);
    }

    r_squared = 1 - (SSres / SStot);
    r = sqrt(r_squared);
    Sy_estimate = sqrt(Syx / (n - 3));

    cout << "Sy (Standard Deviation of y): " << sqrt(Sy / (n - 1)) << endl;
    cout << "Syx (Standard Error of Estimate): " << Sy_estimate << endl;
    cout << "r (Correlation Coefficient): " << r << endl;
    cout << "r^2 (Coefficient of Determination): " << r_squared << endl;
}

int main() {
    vector<double> x = {0, 4, 8, 12, 16, 20};
    vector<double> y = {67, 84, 98, 125, 149, 185};

    double a0, a1, a2;
    quadraticRegression(x, y, a0, a1, a2);

    cout << "Quadratic Regression Coefficients:" << endl;
    cout << "a0 = " << a0 << endl;
    cout << "a1 = " << a1 << endl;
    cout << "a2 = " << a2 << endl;

    double prediction = predict(a0, a1, a2, 40);
    cout << "Predicted cell concentration after 40 days: " << prediction << " million cells" << endl;

    calculateMetrics(x, y, a0, a1, a2);

    return 0;
}

