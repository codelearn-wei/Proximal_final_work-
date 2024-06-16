#pragma once
#include <vector>
#include "ProximalOperator.h"

class L2NormProx : public ProximalOperator {
    double lambda;

public:
    L2NormProx(double lambda);
    std::vector<double> apply(const std::vector<double>& x, double t) const override;
    double evaluate(const std::vector<double>& x) const override;
};
