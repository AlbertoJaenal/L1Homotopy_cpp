#ifndef BDPN_HOMOTOPY_HPP_
#define BDPN_HOMOTOPY_HPP_

#include "Solver_homotopy.hpp"
#include <eigen3/Eigen/Core>
#include <vector>

class BPDNHomotopy : public SolverHomotopy
{

    inline int sign(double a)			{return (a > 0.0)? 1: -1;}
	inline int min_wrapper(int a, int b){return a <= b ? a : b;}
	
    Eigen::VectorXd xk_temp;
	Eigen::VectorXd del_x_vec;
	Eigen::VectorXd tempD;
	
	// Primal and dual sign and support
	Eigen::VectorXi z_x;  //Primal and dual sign
	std::vector<int> gamma_x; //Primal and dual support
    
	Eigen::VectorXd x_k;

	double epsilon;
	int m, n;

	double threshold;
	int maxIter;
    bool verbose;
    
public:
	BPDNHomotopy() : threshold(0.01), maxIter(100), verbose(false){}

	BPDNHomotopy(double thres, int mIter) : maxIter(mIter), threshold(thres)
		{verbose = false;}

	BPDNHomotopy(double thres, int mIter, bool verb) : threshold(thres), maxIter(mIter), verbose(verb)
    {}

	void solveHomotopy(const Eigen::VectorXd &y, const Eigen::MatrixXd &A, Eigen::VectorXd& xk_1);
private:
	void  update_primal(int &out_xi, double &delta, int &i_delta, const Eigen::VectorXd &pk, const Eigen::VectorXd &dk);
	void  update_inverse( const Eigen::MatrixXd &AtB, const Eigen::MatrixXd &iAtB_old, Eigen::MatrixXd &iAtB, int flag);
};

#endif /* BDPN_HOMOTOPY_HPP_ */
