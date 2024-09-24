#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

struct RegressionResult {
    double a, b;  // y = a + bx
    double r2;    // Coefficient of determination
    double Sy;    // Standard deviation of y
    double Syx;   // Standard error of estimate
    double r;     // Correlation coefficient
};

RegressionResult linearRegression(const vector<double>& x, const vector<double>& y) {
    int n = x.size();
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

    // Sum all x, y, x*y, and x^2
    for (int i = 0; i < n; ++i) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
    }

    // Calculate slope (b) and intercept (a)
    double b = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    double a = (sumY - b * sumX) / n;

    // Calculate mean of y
    double meanY = sumY / n;

    // Calculate total sum of squares (SST) and residual sum of squares (SSR)
    double ssTot = 0, ssRes = 0;
    for (int i = 0; i < n; ++i) {
        double yPred = a + b * x[i];  // Predicted y using the model
        ssRes += pow(y[i] - yPred, 2);  // Residual sum of squares
        ssTot += pow(y[i] - meanY, 2);  // Total sum of squares
    }

    // Calculate coefficient of determination (r^2)
    double r2 = 1 - (ssRes / ssTot);

    // Standard deviation of y (Sy)
    double Sy = sqrt(ssTot / (n - 1));

    // Standard error of estimate (Sy/x)
    double Syx = sqrt(ssRes / (n - 2));

    // Correlation coefficient (r)
    double r = sqrt(r2);

    return {a, b, r2, Sy, Syx, r};
}

int main() {
    // Input data: x = days, y = cell count
    vector<double> x = {0, 4, 8, 12, 16, 20};
    vector<double> y = {67, 84, 98, 125, 149, 185};

    // Perform linear regression
    RegressionResult result = linearRegression(x, y);

    // Display results
    cout << "Linear Regression: y = " << result.a << " + " << result.b << "x\n";
    cout << "R^2 (Coefficient of Determination) = " << result.r2 << endl;
    cout << "Standard Deviation of y (Sy) = " << result.Sy << endl;
    cout << "Standard Error of Estimate (Sy/x) = " << result.Syx << endl;
    cout << "Correlation Coefficient (r) = " << result.r << endl;

    // Prediction for 40 days
    double prediction = result.a + result.b * 40;
    cout << "Prediction for 40 days: " << prediction << endl;

    return 0;
}
