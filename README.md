# L1Homotopy++ 
===========

This repository contains a C++/Eigen3 implementation of the L1-norm minimization using homotopy.

**Related Publications:** 
        - Asif, M. S., & Romberg, J. (2010). Dynamic Updating for l1 Minimization. IEEE Journal of selected topics in signal processing, 4(2), 421-434. [arXiv](https://arxiv.org/abs/0903.1443)

**Original code (MATLAB):** [L1-homotopy/Pursuits_Homotopy](https://github.com/sasif/L1-homotopy/tree/master/Pursuits_Homotopy)

## 1. Dependencies

* Eigen (3.3~beta1-2)
   ```
    sudo apt install libeigen3-dev
   ```

A .so library will be created in **lib/** for its usage in bigger projects.

## 2. Overview of the algorithm
At the current state of the repository, only the **Dantzig selector (DS) homotopy based on primal-dual pursuit** [(DS_homotopy)]() has been implemented.

The problem is considered as a linear model of observations: 
   ```
    y = Ax+e
   ```
where x is a sparse vector of interest, A is the system matrix, y is the measurement vector, and e is the noise.   
We want to solve the following weighted L-norm minimization program:  
```
	minimize_x  ||x||_1  s.t 1/2*||Ax-y||_2^2,  
```
as
```
	minimize_x  tau * ||x||_1  + 1/2*||Ax-y||_2^2,  
```
  
## 3. Usage

The class must be initialized with the tolerance, the maximum number of iterations and the verbose:
   ```
    std::unique_ptr<DSHomotopy> solver;
	solver.reset( new DSHomotopy(1e-4, 100, false));
   ```
The L1Homotopy algorithm solves the optimization problem through the next method:
   ```
    solver->solveHomotopy(const Eigen::VectorXd &y, const Eigen::MatrixXd &A, Eigen::VectorXd& xk_1);
   ```
where xk_1 is the solution, A is the system matrix, y is the measurement vector. 

The experimental fucntion:
   ```
    solver->solveHomotopy_primal(const Eigen::VectorXd &y, const Eigen::MatrixXd &A, Eigen::VectorXd& xk_1, double search_lambda);
   ```
solves the same problem speeding it up by testing the residual of the function until it falls below certain the class tolerance.

### Tests

Test examples are given in **test/** with some MATLAB-generated data (**test/data/**) in order to test the performance of the algorithm.
