#pragma once
#include <vector>

class ProximalOperator {
public:
    virtual ~ProximalOperator() {}
    virtual std::vector<double> apply(const std::vector<double>& x, double t) const = 0;
    virtual double evaluate(const std::vector<double>& x) const = 0;
};
