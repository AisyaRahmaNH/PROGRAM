#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

struct QuadraticResult {
    double a, b, c; // y = a + bx + cx^2
    double r2;
};

QuadraticResult quadraticRegression(const vector<double>& x, const vector<double>& y) {
    int n = x.size();
    double sumX = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0, sumY = 0, sumXY = 0, sumX2Y = 0;

    for (int i = 0; i < n; ++i) {
        sumX += x[i];
        sumX2 += pow(x[i], 2);
        sumX3 += pow(x[i], 3);
        sumX4 += pow(x[i], 4);
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2Y += pow(x[i], 2) * y[i];
    }

    double denominator = n * (sumX2 * sumX4 - pow(sumX3, 2)) - sumX * (sumX * sumX4 - sumX2 * sumX3) + sumX2 * (sumX * sumX3 - sumX2 * sumX2);
    
    double a = (sumY * (sumX2 * sumX4 - pow(sumX3, 2)) - sumX * (sumXY * sumX4 - sumX2Y * sumX3) + sumX2 * (sumXY * sumX3 - sumX2Y * sumX2)) / denominator;
    double b = (n * (sumXY * sumX4 - sumX2Y * sumX3) - sumY * (sumX * sumX4 - sumX2 * sumX3) + sumX2 * (sumX * sumX2Y - sumXY * sumX2)) / denominator;
    double c = (n * (sumX2 * sumX2Y - sumXY * sumX3) - sumX * (sumX * sumX2Y - sumXY * sumX3) + sumY * (sumX * sumX3 - sumX2 * sumX2)) / denominator;

    // Calculate R^2
    double ssTot = 0, ssRes = 0, meanY = sumY / n;
    for (int i = 0; i < n; ++i) {
        double yPred = a + b * x[i] + c * pow(x[i], 2);
        ssRes += pow(y[i] - yPred, 2);
        ssTot += pow(y[i] - meanY, 2);
    }
    double r2 = 1 - (ssRes / ssTot);

    return {a, b, c, r2};
}

int main() {
    vector<double> x = {0, 4, 8, 12, 16, 20};
    vector<double> y = {67, 84, 98, 125, 149, 185};

    QuadraticResult result = quadraticRegression(x, y);

    cout << "Quadratic Regression: y = " << result.a << " + " << result.b << "x + " << result.c << "x^2\n";
    cout << "R^2 = " << result.r2 << endl;

    // Prediction for 40 days
    double prediction = result.a + result.b * 40 + result.c * pow(40, 2);
    cout << "Prediction for 40 days: " << prediction << endl;

    return 0;
}
