# L1Homotopy++ 
===========

This repository contains a C++/Eigen3 implementation of the L1-norm minimization using homotopy, concretely Primal-Dual pursuit approaches.

**Related Publications:** 

* Asif, M. Salman, and Justin Romberg. "Dynamic Updating for $\ell_ {1} $ Minimization." IEEE Journal of selected topics in signal processing 4.2 (2010): 421-434. [arXiv](https://arxiv.org/abs/0903.1443)
	
* Asif, Muhammad Salman. Primal Dual Pursuit: A homotopy based algorithm for the Dantzig selector. Diss. Georgia Institute of Technology, 2008. [pdf](http://130.203.136.95/viewdoc/download;jsessionid=97DAA0421D7545BD6E73406C6765461A?doi=10.1.1.329.2692&rep=rep1&type=pdf)

**Original GitHub repo (MATLAB):** [L1-homotopy/Pursuits_Homotopy](https://github.com/sasif/L1-homotopy/tree/master/Pursuits_Homotopy)

**Original repo:** [
  https://intra.ece.ucr.edu/~sasif/homotopy/index.html](
  https://intra.ece.ucr.edu/~sasif/homotopy/index.html)
  
**License of the authors**: [here](https://github.com/sasif/L1-homotopy/blob/master/license.txt)

## 1. Dependencies

* Eigen (3.3~beta1-2)
   ```
    sudo apt install libeigen3-dev
   ```

A .so library will be created in **lib/** if you want to include them in bigger projects.

## 2. Overview of the algorithm
At the current state of the repository, implementations are given for the **Dantzig selector (DS) homotopy based on primal-dual pursuit** or the **Basis pursuit denoising (BPDN) homotopy**.

The problem is considered as a linear model of observations: 
   ```
    y = Ax+e
   ```
where x is a sparse n-vector, A is the system mxn-matrix, y is the measurement m-vector, and e is the noise.   
We want to solve the following weighted L1-norm minimization program:  
```
	minimize_x  ||x||_1  s.t ||Ax-y||_2^2 < \eps,  
```

The DS algorithm performs a primal dual optimization, while the BPDN algorithm forces that the support of the primal and dual vectors remain same at every step.

## 3. Usage

Both algorithms must be initialized as classes with the tolerance, the maximum number of iterations and the verbose:
   ```
    std::unique_ptr<SolverHomotopy> solver;
    solver.reset( new DSHomotopy(1e-4, 100, false));
    or 
    solver.reset( new BPDNHomotopy(0.1, 100, false));
   ```
The SolverHomotopy class solves the optimization problem through the next method:
   ```
    solver->solveHomotopy(const Eigen::VectorXd &y, const Eigen::MatrixXd &A, Eigen::VectorXd& xk_1);
   ```
where xk_1 is the solution, A is the system matrix, y is the measurement vector.

### Tests

Test examples are given in **test/** with some MATLAB-generated data (**test/data/**) in order to test the performance of the algorithm.
