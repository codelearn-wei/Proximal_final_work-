#include "ProximalGradientOptimizer.h"
#include <cmath>
#include <numeric>
#include <iostream>
#include <fstream>

 void saveToCSV(const std::string& filename, const std::vector<double>& data) {
        std::ofstream file(filename);
        file << "Iteration,GradientNorm\n";
        for (size_t i = 0; i < data.size(); ++i) {
            file << i + 1 << "," << data[i] << "\n";
        }
        file.close();
    }
ProximalGradientOptimizer::ProximalGradientOptimizer(Function* f, ProximalOperator* g, double initial_step_size, int max_iterations, bool use_BB_step)
    : f(f), g(g), initial_step_size(initial_step_size), max_iterations(max_iterations), use_BB_step(use_BB_step) {}

double tolerance = 3e-2; // 梯度范数的容忍度

std::vector<double> ProximalGradientOptimizer::optimize(const std::vector<double>& initial) {
    std::vector<double> x = initial, x_new;
    double t = initial_step_size;
    iterations = 0;
     grad_norms.clear(); // 清除历史数据

    for (int iter = 0; iter < max_iterations; ++iter) {
        std::vector<double> grad = f->gradient(x);
        double grad_norm = std::sqrt(std::inner_product(grad.begin(), grad.end(), grad.begin(), 0.0));
        // std::cout<<"fanshu:"<<grad_norm;
        grad_norms.push_back(grad_norm); // 存储梯度范数

        if (grad_norm < tolerance) { // 检查梯度范数是否足够小
            break;
        }

        if (use_BB_step && !x_prev.empty()) {
            t = barzilai_borwein_step(x, grad);
        } else {
            t = armijo_backtracking(x, grad, t);
        }

        x_new = x;
        for (size_t i = 0; i < x.size(); ++i) {
            x_new[i] = x[i] - t * grad[i];
        }
        x_new = g->apply(x_new, t);

        // 更新变量准备下一次迭代
        grad_prev = grad;
        x_prev = x;
        x = x_new;

        iterations++; // 迭代次数加1
        //saveToCSV("PGM/data_ansys/grad_data/L1_Ls1_bb_big.csv", grad_norms);
    }
    return x;
}

double ProximalGradientOptimizer::armijo_backtracking(const std::vector<double>& x, const std::vector<double>& grad, double t, double alpha, double beta) {
    double f_x = f->evaluate(x); // 初始点的函数值
    std::vector<double> x_new(x.size()); // 新点的容器
    double f_x_new;

    do {
        // 使用当前步长更新 x_new
        for (size_t i = 0; i < x.size(); ++i) {
            x_new[i] = x[i] - t * grad[i];
        }
        f_x_new = f->evaluate(x_new); // 新点的函数值

        // 检查Armijo条件
        if (f_x_new <= f_x - alpha * t * std::inner_product(grad.begin(), grad.end(), grad.begin(), 0.0)) {
            break; // 如果满足条件，则退出循环
        }

        t *= beta; // 否则减小步长
    } while (true); // 保证至少运行一次，但通常需要有退出条件以防止无限循环

    return t; // 返回满足Armijo条件的步长
}


double ProximalGradientOptimizer::barzilai_borwein_step(const std::vector<double>& x, const std::vector<double>& grad) {
    if (x_prev.empty()) {
        // 第一次迭代没有前一步的数据，使用一个合适的默认步长
        return initial_step_size;
    }

    // 计算 s 和 y
    std::vector<double> s(x.size()), y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        s[i] = x[i] - x_prev[i];
        y[i] = grad[i] - grad_prev[i];
    }

    double sy = std::inner_product(s.begin(), s.end(), y.begin(), 0.0);
    double ss = std::inner_product(s.begin(), s.end(), s.begin(), 0.0);

    if (sy == 0) {
        return std::max(initial_step_size, 1e-8);  // 避免步长为零
    }

    double step = ss / sy; // 基本的 BB 步长
    double alphaMax = 1e5;
    double alphaMin = 1e-5;

    // 约束步长范围
    step = std::max(std::min(step, alphaMax), alphaMin);

    // 线搜索过程（模拟的简化示例）
    double alpha = 0.5;
    int nls = 0, max_nls = 10;  // 线搜索迭代次数限制
    double f_x = f->evaluate(x);  // 当前 x 的函数值
    std::vector<double> x_new(x.size());

    while (nls < max_nls) {
        for (size_t i = 0; i < x.size(); ++i) {
            x_new[i] = x[i] - step * grad[i];
        }
        double f_x_new = f->evaluate(x_new);

        if (f_x_new < f_x - alpha * step * std::inner_product(grad.begin(), grad.end(), grad.begin(), 0.0)) {
            break;  // 符合Armijo条件
        }

        step *= 0.5;  // 步长减半
        nls++;
    }

    return step;
}

