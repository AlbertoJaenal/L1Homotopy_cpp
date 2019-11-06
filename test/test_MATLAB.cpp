/*

This function is currently not in use, but it has been used in the debugging of
the algorithm.

*/

#include <iostream>
#include <fstream>
#include <memory>

#include "Solver_homotopy.hpp"
#include "DS_homotopy.h"
#include "BPDN_homotopy.h"

using namespace std;

Eigen::MatrixXd readCSV(std::string file, int rows, int cols) {

	std::ifstream in(file);

	std::string line;

	int row = 0;
	int col = 0;

	Eigen::MatrixXd res = Eigen::MatrixXd(rows, cols);

	if (in.is_open()) {

	while (std::getline(in, line)) {
		char *ptr = (char *) line.c_str();
		int len = line.length();

		col = 0;

		char *start = ptr;
		for (int i = 0; i < len; i++) 
		{

			if (ptr[i] == ',') 
			{
				res(row, col++) = atof(start);
				start = ptr + i + 1;
			}
		}
		res(row, col) = atof(start);

		row++;
	}

	in.close();
  }
  return res;
}


Eigen::VectorXd readCSV(std::string file, int rows) {

	std::ifstream in(file);

	std::string line;

	int row = 0;

	Eigen::VectorXd res(rows);

	if (in.is_open()) 
	{
		while (std::getline(in, line)) 
		{
			char *ptr = (char *) line.c_str();
			char *start = ptr;
			res(row) = atof(start);

			row++;
		}

		in.close();
	}
	return res;
}


int main(int argc, char *argv[]) 
{
	
	const size_t M = 250;
	const size_t N = 512;
	Eigen::MatrixXd A = readCSV("../test/data/A_MATLAB.csv", M, N);
	Eigen::VectorXd x  = readCSV("../test/data/x_MATLAB.csv", N);
	Eigen::VectorXd b = readCSV("../test/data/y_MATLAB.csv", M);
    
	Eigen::VectorXd X2;
    
    
	std::unique_ptr<SolverHomotopy> solver;
	solver.reset( new DSHomotopy(1e-4, 100, true));
	solver->solveHomotopy(b, A, X2);    
    std::cout << "\nThe norm of the error with DS is " << (X2 - x).norm();
    
	solver.reset( new BPDNHomotopy(1e-4, 10000, true));
	solver->solveHomotopy(b, A, X2);    
    std::cout << "\nThe norm of the error with BPDN is " << (X2 - x).norm();
	
	return -1;
}











