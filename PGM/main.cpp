#include <iostream>
#include <vector>
#include <random>
#include "LeastSquares.h"
#include "LogisticRegression.h"
#include "L1NormProx.h"
#include "L2NormProx.h"
#include "L0NormProx.h"
#include "ProximalGradientOptimizer.h"
#include <chrono>



void generateRandomVector(std::vector<double>& vec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0);

    for (double& element : vec) {
        element = dis(gen);
    }
}

void generateRandomMatrix(std::vector<std::vector<double>>& mat) {
    for (auto& row : mat) {
        generateRandomVector(row);
    }
}

int main() {
    const size_t rows = 512;
    const size_t cols = 512;
    // big
    std::vector<std::vector<double>> A(rows, std::vector<double>(cols));
    std::vector<double> b(rows);
    generateRandomMatrix(A);
    generateRandomVector(b);
    std::vector<double> x(cols, 0.0); // initial guess

    // small
    // std::vector<std::vector<double>> A = {
    //     {1, 3},
    //     {2, 1}
    // };
    // std::vector<double> b = {{0}, {2}};
    // std::vector<double>x = {{1000},{2000}};

 // 设置要测试的配置
    std::string modelType = "LR"; // Or "LR"
    std::string normType = "L0"; // Or "L2", "L0"
    bool useBBStep = false; // Or false

    Function* f = nullptr;
    ProximalOperator* g = nullptr;
    ProximalGradientOptimizer* optimizer = nullptr;

    // Function choice
    if (modelType == "LS") {
        f = new LeastSquares(A, b);
    } else if (modelType == "LR") {
        f = new LogisticRegression(A, b);
    }

    // Norm choice
    if (normType == "L1") {
        g = new L1NormProx(0.001);
    } else if (normType == "L2") {
        g = new L2NormProx(0.001);
    } else if (normType == "L0") {
        g = new L0NormProx(0.001);
    }

    optimizer = new ProximalGradientOptimizer(f, g, 1, 3000, useBBStep);


  
    // Perform optimization
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<double> solution = optimizer->optimize(x);
    auto end = std::chrono::high_resolution_clock::now();
    double time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

    // Compute the objective value
    double obj_val = f->evaluate(solution) + g->evaluate(solution);


 // Print results
    std::cout << "Iterations: " << optimizer->getIterations() << std::endl;
    std::cout << "Objective Value: " << obj_val << std::endl;
    std::cout << "Solution: [ ";
    for (auto &val : solution) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
    std::cout << "CPU Time: " << time_taken << " s" << std::endl;

    delete f;
    delete g;

    return 0;
}
