#include "LeastSquares.h"
#include <cmath>

LeastSquares::LeastSquares(const std::vector<std::vector<double>>& A, const std::vector<double>& b) : A(A), b(b) {}

double LeastSquares::evaluate(const std::vector<double>& x) const {
    double sum = 0.0;
    for (size_t i = 0; i < A.size(); ++i) {
        double Ax_b = 0.0;
        for (size_t j = 0; j < x.size(); ++j) {
            Ax_b += A[i][j] * x[j];
        }
        Ax_b -= b[i];
        sum += Ax_b * Ax_b;
    }
    return 0.5 * sum;
}

std::vector<double> LeastSquares::gradient(const std::vector<double>& x) const {
    std::vector<double> grad(x.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i) {
        double Ax_b = 0.0;
        for (size_t j = 0; j < x.size(); ++j) {
            Ax_b += A[i][j] * x[j];
        }
        Ax_b -= b[i];
        for (size_t j = 0; j < x.size(); ++j) {
            grad[j] += 2 * A[i][j] * Ax_b;
        }
    }
    return grad;
}
