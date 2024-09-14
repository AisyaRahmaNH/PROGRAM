#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Function prototypes
void printMatrix(const vector<vector<double>>& matrix);
void jacobiIteration(const vector<vector<double>>& A, const vector<double>& b, vector<double>& x, int iterations, double tolerance);
void naiveGaussianElimination(vector<vector<double>>& A, vector<double>& b);
double determinantNaiveGaussianElimination(vector<vector<double>> A);
void gaussJordanElimination(vector<vector<double>>& A, vector<double>& b);

int main() {
    // Problem data
    vector<vector<double>> A = {
        {15, -3, -1},
        {-3, 18, -6},
        {-4, -1, 12}
    };
    vector<double> b = {3800, 1200, 2350};
    vector<double> x(3, 0.0); // Solution vector for Jacobi iteration
    
    // Jacobi Iteration
    cout << "Jacobi Iteration:\n";
    jacobiIteration(A, b, x, 25, 0.05);
    for (double xi : x) cout << xi << " ";
    cout << endl;
    
    // Naive Gaussian Elimination with Back Substitution
    vector<vector<double>> A1 = A;
    vector<double> b1 = b;
    cout << "Naive Gaussian Elimination with Back Substitution:\n";
    naiveGaussianElimination(A1, b1);
    for (double xi : b1) cout << xi << " ";
    cout << endl;
    
    // Determinant using Naive Gaussian Elimination
    vector<vector<double>> A2 = A;
    cout << "Determinant using Naive Gaussian Elimination:\n";
    double determinant = determinantNaiveGaussianElimination(A2);
    cout << determinant << endl;

    // Gauss-Jordan Elimination
    vector<vector<double>> A3 = A;
    vector<double> b3 = b;
    cout << "Gauss-Jordan Elimination:\n";
    gaussJordanElimination(A3, b3);
    for (double xi : b3) cout << xi << " ";
    cout << endl;

    return 0;
}

void printMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double value : row) {
            cout << value << " ";
        }
        cout << endl;
    }
}

void jacobiIteration(const vector<vector<double>>& A, const vector<double>& b, vector<double>& x, int iterations, double tolerance) {
    int n = A.size();
    vector<double> x_old(n, 0.0);
    for (int it = 0; it < iterations; ++it) {
        x_old = x;
        for (int i = 0; i < n; ++i) {
            double sum = b[i];
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    sum -= A[i][j] * x_old[j];
                }
            }
            x[i] = sum / A[i][i];
        }
        double error = 0.0;
        for (int i = 0; i < n; ++i) {
            error = max(error, fabs(x[i] - x_old[i]));
        }
        if (error < tolerance) {
            break;
        }
    }
}

void naiveGaussianElimination(vector<vector<double>>& A, vector<double>& b) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        // Partial pivoting
        int maxRow = i;
        for (int k = i + 1; k < n; ++k) {
            if (fabs(A[k][i]) > fabs(A[maxRow][i])) {
                maxRow = k;
            }
        }
        if (i != maxRow) {
            swap(A[i], A[maxRow]);
            swap(b[i], b[maxRow]);
        }

        // Forward Elimination
        for (int k = i + 1; k < n; ++k) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; ++j) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }
    // Back Substitution
    for (int i = n - 1; i >= 0; --i) {
        b[i] /= A[i][i];
        for (int k = i - 1; k >= 0; --k) {
            b[k] -= A[k][i] * b[i];
        }
    }
}

double determinantNaiveGaussianElimination(vector<vector<double>> A) {
    int n = A.size();
    double det = 1.0;
    for (int i = 0; i < n; ++i) {
        // Partial pivoting
        int maxRow = i;
        for (int k = i + 1; k < n; ++k) {
            if (fabs(A[k][i]) > fabs(A[maxRow][i])) {
                maxRow = k;
            }
        }
        if (i != maxRow) {
            swap(A[i], A[maxRow]);
            det *= -1;
        }
        
        if (A[i][i] == 0) return 0; // Singular matrix
        
        det *= A[i][i];
        
        for (int k = i + 1; k < n; ++k) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; ++j) {
                A[k][j] -= factor * A[i][j];
            }
        }
    }
    return det;
}

void gaussJordanElimination(vector<vector<double>>& A, vector<double>& b) {
    int n = A.size();
    vector<vector<double>> augmented(n, vector<double>(n + 1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmented[i][j] = A[i][j];
        }
        augmented[i][n] = b[i];
    }
    
    // Forward elimination to get RREF
    for (int i = 0; i < n; ++i) {
        double diag = augmented[i][i];
        for (int j = 0; j < n + 1; ++j) {
            augmented[i][j] /= diag;
        }
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = augmented[k][i];
                for (int j = 0; j < n + 1; ++j) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }
    
    // Extract solution
    for (int i = 0; i < n; ++i) {
        b[i] = augmented[i][n];
    }
}
