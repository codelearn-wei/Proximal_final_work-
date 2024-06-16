#pragma once
#include <vector>

class Function {
public:
    virtual double evaluate(const std::vector<double>& x) const = 0;
    virtual std::vector<double> gradient(const std::vector<double>& x) const = 0;
    virtual ~Function() {}
};
