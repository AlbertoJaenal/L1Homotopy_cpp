#ifndef SOLVER_HOMOTOPY_HPP_
#define SOLVER_HOMOTOPY_HPP_

#include <eigen3/Eigen/Core>

class SolverHomotopy
{
public:
	virtual void solveHomotopy(const Eigen::VectorXd &y, const Eigen::MatrixXd &A, Eigen::VectorXd& xk_1){};
	virtual ~SolverHomotopy(){}
};


#endif /* SOLVER_HOMOTOPY_HPP_ */
