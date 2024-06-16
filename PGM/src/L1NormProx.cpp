#include "L1NormProx.h"
#include <cmath>  // 引入用于计算绝对值的头文件

L1NormProx::L1NormProx(double lambda) : lambda(lambda) {}

std::vector<double> L1NormProx::apply(const std::vector<double>& x, double t) const {
    std::vector<double> result(x.size());
    double thresh = lambda * t;
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] > thresh) {
            result[i] = x[i] - thresh;
        } else if (x[i] < -thresh) {
            result[i] = x[i] + thresh;
        } else {
            result[i] = 0;
        }
    }
    return result;
}

double L1NormProx::evaluate(const std::vector<double>& x) const {
    double sum = 0.0;
    for (auto val : x) {
        sum += std::abs(val);  // 计算所有元素的绝对值和
    }
    return sum;
}
