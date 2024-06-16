#include "L0NormProx.h"
#include <cmath>  // 引入用于计算绝对值的头文件

L0NormProx::L0NormProx(double lambda) : lambda(lambda) {}

std::vector<double> L0NormProx::apply(const std::vector<double>& x, double t) const {
    std::vector<double> result(x.size());
    double threshold = lambda * t; // 设定阈值，通常与步长和lambda参数有关

    for (size_t i = 0; i < x.size(); ++i) {
        // 硬阈值处理：只保留绝对值大于阈值的元素
        if (fabs(x[i]) > threshold) {
            result[i] = x[i];
        } else {
            result[i] = 0;
        }
    }
    return result;
}

double L0NormProx::evaluate(const std::vector<double>& x) const {
    int non_zero_count = 0;
    for (auto value : x) {
        if (value != 0) {
            non_zero_count++;  // 对非零元素进行计数
        }
    }
    return static_cast<double>(non_zero_count);  // 返回非零元素的数量
}
