#pragma once
#include "ProximalOperator.h"
#include <vector>

class L1NormProx : public ProximalOperator {
    double lambda;

public:
    L1NormProx(double lambda);
    std::vector<double> apply(const std::vector<double>& x, double t) const override;
    double evaluate(const std::vector<double>& x) const override;
};
