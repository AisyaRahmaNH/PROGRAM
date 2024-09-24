#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

struct RegressionResult {
    double a, b; // y = a + bx
    double r2;
};

RegressionResult linearRegression(const vector<double>& x, const vector<double>& y) {
    int n = x.size();
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    
    for (int i = 0; i < n; ++i) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
    }
    
    double b = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    double a = (sumY - b * sumX) / n;
    
    // Calculate R^2
    double ssTot = 0, ssRes = 0, meanY = sumY / n;
    for (int i = 0; i < n; ++i) {
        double yPred = a + b * x[i];
        ssRes += pow(y[i] - yPred, 2);
        ssTot += pow(y[i] - meanY, 2);
    }
    double r2 = 1 - (ssRes / ssTot);
    
    return {a, b, r2};
}

int main() {
    vector<double> x = {0, 4, 8, 12, 16, 20};
    vector<double> y = {67, 84, 98, 125, 149, 185};
    
    RegressionResult result = linearRegression(x, y);
    
    cout << "Linear Regression: y = " << result.a << " + " << result.b << "x\n";
    cout << "R^2 = " << result.r2 << endl;

    // Prediction for 40 days
    double prediction = result.a + result.b * 40;
    cout << "Prediction for 40 days: " << prediction << endl;

    return 0;
}
