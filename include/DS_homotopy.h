#ifndef DS_HOMOTOPY_HPP_
#define DS_HOMOTOPY_HPP_

#include <eigen3/Eigen/Core>
#include <vector>

class DSHomotopy
{

    inline int sign(double a)			{return (a > 0.0)? 1: -1;}
	inline int min_wrapper(int a, int b){return a <= b ? a : b;}
	
    Eigen::VectorXd xk_temp, lambdak_temp;
	Eigen::VectorXd del_x_vec, del_lambda_p;
	Eigen::VectorXd tempD;
	
	// Primal and dual sign and support
	Eigen::VectorXi z_x, z_lambda;  //Primal and dual sign
	std::vector<int> gamma_x, gamma_lambda; //Primal and dual support
    
	Eigen::VectorXd x_k;
	Eigen::VectorXd lambda_k;

	double epsilon;
	int nIter;

    bool verbose;

	// global N gamma_x z_x  xk_temp del_x_vec pk_temp dk epsilon isNonnegative


	double tolerance;
	int maxIter;
	int m, n;
	
public:
	DSHomotopy() : tolerance(0.001), maxIter(100), verbose(false){}

	DSHomotopy(double tol, int mIter) : tolerance(tol), maxIter(mIter)
		{verbose = false;}

	DSHomotopy(double tol, int mIter, bool verb) : tolerance(tol), maxIter(mIter), verbose(verb)
    {}

    void solveHomotopy(const Eigen::VectorXd &y, const Eigen::MatrixXd &A, Eigen::VectorXd& xk_1);
	void solveHomotopy_primal(const Eigen::VectorXd &y, const Eigen::MatrixXd &A, Eigen::VectorXd& xk_1, double search_lambda);
private:
	void  update_primal(int &out_xi, double &delta, int &i_delta, const Eigen::VectorXd &pk, const Eigen::VectorXd &dk);
	void  update_dual(int &out_lambdai, double &theta, int &i_theta, const Eigen::VectorXd &ak, const Eigen::VectorXd &bk, int new_lambda);
	void update_inverse( const Eigen::MatrixXd &AtB, const Eigen::MatrixXd &iAtB_old, Eigen::MatrixXd &iAtB, int flag);
};

#endif /* HOMOTOPY_HPP_ */
