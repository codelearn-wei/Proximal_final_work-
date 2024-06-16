#include "LogisticRegression.h"

LogisticRegression::LogisticRegression(const std::vector<std::vector<double>>& A, const std::vector<double>& b) : A(A), b(b) {}

double LogisticRegression::evaluate(const std::vector<double>& x) const {
    double loss = 0.0;
    for (size_t i = 0; i < A.size(); ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < x.size(); ++j) {
            dot += A[i][j] * x[j];
        }
        loss += log(1 + exp(-b[i] * dot));
    }
    return loss;
}

std::vector<double> LogisticRegression::gradient(const std::vector<double>& x) const {
    std::vector<double> grad(x.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < x.size(); ++j) {
            dot += A[i][j] * x[j];
        }
        double exp_term = exp(-b[i] * dot);
        double factor = (1 / (1 + exp_term) - 1) * b[i];
        for (size_t j = 0; j < x.size(); ++j) {
            grad[j] += A[i][j] * factor;
        }
    }
    for (size_t j = 0; j < x.size(); ++j) {
        grad[j] /= A.size(); // Normalize by the number of samples m
    }
    return grad;
}