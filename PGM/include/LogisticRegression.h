// LogisticRegression.h
#include "Function.h"
#include <cmath>
#include <vector>

class LogisticRegression : public Function {
    std::vector<std::vector<double>> A;
    std::vector<double> b;

public:
    LogisticRegression(const std::vector<std::vector<double>>& A, const std::vector<double>& b);
    double evaluate(const std::vector<double>& x) const override;
    std::vector<double> gradient(const std::vector<double>& x) const override;
};