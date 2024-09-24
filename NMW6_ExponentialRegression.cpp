#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// Function to calculate the exponential regression coefficients
void exponentialRegression(const vector<double>& x, const vector<double>& y, double& a, double& b) {
    int n = x.size();
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    vector<double> lnY(n);
    
    // Convert y to ln(y)
    for (int i = 0; i < n; ++i) {
        lnY[i] = log(y[i]);
        sumX += x[i];
        sumY += lnY[i];
        sumXY += x[i] * lnY[i];
        sumX2 += x[i] * x[i];
    }
    
    b = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    a = exp((sumY - b * sumX) / n);
}

// Function to predict y using the exponential model y = a * exp(b * x)
double predict(double a, double b, double x) {
    return a * exp(b * x);
}

// Function to compute statistical metrics
void calculateMetrics(const vector<double>& x, const vector<double>& y, double a, double b) {
    int n = x.size();
    double Sy = 0, Syx = 0, mean_y = 0, sum_error = 0, SStot = 0, SSres = 0, r_squared, r, Sy_estimate;

    for (int i = 0; i < n; ++i) {
        mean_y += y[i];
    }
    mean_y /= n;

    for (int i = 0; i < n; ++i) {
        double y_pred = predict(a, b, x[i]);
        SStot += pow(y[i] - mean_y, 2);
        SSres += pow(y[i] - y_pred, 2);
        Sy += pow(y[i] - mean_y, 2);
        Syx += pow(y[i] - y_pred, 2);
    }

    r_squared = 1 - (SSres / SStot);
    r = sqrt(r_squared);
    Sy_estimate = sqrt(Syx / (n - 2));

    cout << "Sy (Standard Deviation of y): " << sqrt(Sy / (n - 1)) << endl;
    cout << "Syx (Standard Error of Estimate): " << Sy_estimate << endl;
    cout << "r (Correlation Coefficient): " << r << endl;
    cout << "r^2 (Coefficient of Determination): " << r_squared << endl;
}

int main() {
    vector<double> x = {0, 4, 8, 12, 16, 20};
    vector<double> y = {67, 84, 98, 125, 149, 185};

    double a, b;
    exponentialRegression(x, y, a, b);

    cout << "Exponential Regression Coefficients:" << endl;
    cout << "A = " << a << endl;
    cout << "B = " << b << endl;

    double prediction = predict(a, b, 40);
    cout << "Predicted cell concentration after 40 days: " << prediction << " million cells" << endl;

    calculateMetrics(x, y, a, b);

    return 0;
}

