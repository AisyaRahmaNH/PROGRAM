#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

struct ExpResult {
    double a, b; // y = a * e^(bx)
    double r2;
};

ExpResult exponentialRegression(const vector<double>& x, const vector<double>& y) {
    int n = x.size();
    double sumX = 0, sumLogY = 0, sumXLogY = 0, sumX2 = 0;

    for (int i = 0; i < n; ++i) {
        sumX += x[i];
        sumLogY += log(y[i]);
        sumXLogY += x[i] * log(y[i]);
        sumX2 += x[i] * x[i];
    }

    double b = (n * sumXLogY - sumX * sumLogY) / (n * sumX2 - sumX * sumX);
    double a = exp((sumLogY - b * sumX) / n);

    // Calculate R^2
    double ssTot = 0, ssRes = 0, meanLogY = sumLogY / n;
    for (int i = 0; i < n; ++i) {
        double logYPred = log(a) + b * x[i];
        ssRes += pow(log(y[i]) - logYPred, 2);
        ssTot += pow(log(y[i]) - meanLogY, 2);
    }
    double r2 = 1 - (ssRes / ssTot);

    return {a, b, r2};
}

int main() {
    vector<double> x = {0, 4, 8, 12, 16, 20};
    vector<double> y = {67, 84, 98, 125, 149, 185};

    ExpResult result = exponentialRegression(x, y);

    cout << "Exponential Regression: y = " << result.a << " * e^(" << result.b << "x)\n";
    cout << "R^2 = " << result.r2 << endl;

    // Prediction for 40 days
    double prediction = result.a * exp(result.b * 40);
    cout << "Prediction for 40 days: " << prediction << endl;

    return 0;
}
