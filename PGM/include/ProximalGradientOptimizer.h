#pragma once
#include "Function.h"
#include "ProximalOperator.h"
#include <vector>
#include <functional>

class ProximalGradientOptimizer {
    Function* f;
    ProximalOperator* g;
    double initial_step_size;
    std::vector<double> x_prev;
    std::vector<double> grad_prev;
    bool use_BB_step; // Flag to use Barzilai-Borwein step size
    int max_iterations;
    int iterations = 0;
    std::vector<double> grad_norms; // 存储每步的梯度范数

public:
    ProximalGradientOptimizer(Function* f, ProximalOperator* g, double initial_step_size, int max_iterations, bool use_BB_step = false);
    std::vector<double> optimize(const std::vector<double>& initial);
    double armijo_backtracking(const std::vector<double>& x, const std::vector<double>& grad, double t, double alpha = 0.01, double beta = 0.8);
    double barzilai_borwein_step(const std::vector<double>& x, const std::vector<double>& grad);
    int getIterations() const { return iterations; }  // 公开方法来获取迭代次数
     const std::vector<double>& getGradientNorms() const { return grad_norms; }
};  
