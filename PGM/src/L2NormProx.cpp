#include "L2NormProx.h"
#include <cmath>  // 引入用于计算开平方根的头文件

L2NormProx::L2NormProx(double lambda) : lambda(lambda) {}

std::vector<double> L2NormProx::apply(const std::vector<double>& x, double t) const {
    double norm = 0.0;
    for (auto xi : x) {
        norm += xi * xi;
    }
    norm = std::sqrt(norm);
    std::vector<double> result(x.size());
    double shrink = std::max(1.0 - lambda * t / norm, 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] * shrink;
    }
    return result;
}

double L2NormProx::evaluate(const std::vector<double>& x) const {
    double norm = 0.0;
    for (auto xi : x) {
        norm += xi * xi;
    }
    return std::sqrt(norm);  // 返回 L_2 范数
}
