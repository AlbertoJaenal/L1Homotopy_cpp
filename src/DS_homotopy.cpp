#include "DS_homotopy.h"

#include <iostream>
#include <iomanip>
#include <numeric>
#include <chrono>


void DSHomotopy::solveHomotopy(const Eigen::VectorXd &y, const Eigen::MatrixXd &A, Eigen::VectorXd& xk_1)
{
    std::chrono::steady_clock::time_point t0, t1, t2;
    double time = 0;
    if (verbose)
    {
        std::cout << std::setprecision(5) << std::fixed;
        t0 = std::chrono::steady_clock::now();
    }
    
    m = A.rows();
    n = A.cols();
    
    // Initialization of primal and dual sign and support
    z_x.resize(0); z_x.setZero(n);
    z_lambda.resize(0); z_lambda.setZero(n);
    gamma_x.clear(); 
    gamma_lambda.clear();  
    
    // Initial step
    Eigen::VectorXd Primal_constrk = -A.transpose() * y;    
    int i; 
    double c = Primal_constrk.array().abs().maxCoeff(&i); 
    
    std::vector<int> gamma_xk = {i};
    std::vector<int> gamma_lambdak = {i};
    if (verbose) {std::cout << "\n\tPrimal Add " << i << "!" << "\tDual Add " << i << "!";}
    
    z_lambda(gamma_lambdak[0]) = sign(Primal_constrk(gamma_lambdak[0]));
    epsilon = c;
    Primal_constrk(gamma_lambdak[0]) = sign(Primal_constrk(gamma_lambdak[0])) * epsilon;
    xk_1.setZero(n); // Final primal Solution
    
    Eigen::VectorXd lambdak_1 = Eigen::VectorXd::Zero(n);
    lambdak_1(gamma_lambdak[0]) = 1 / (A.col(gamma_lambdak[0]).transpose() * A.col(gamma_lambdak[0])) * z_lambda(gamma_lambdak[0]);
    
    x_k.resize(0); x_k.setZero(n);
    lambda_k.resize(0);lambda_k.setZero(n);

    Eigen::VectorXd Dual_constrk = A.transpose() * (A * lambdak_1);
    
    z_x(gamma_xk[0]) = -sign(Dual_constrk(gamma_xk[0]));
    Dual_constrk(gamma_xk[0]) = sign(Dual_constrk(gamma_xk[0]));
    
    Eigen::VectorXi z_xk(z_x);
    Eigen::VectorXi z_lambdak(z_lambda);
    
    // Loop variables
    int iteration = 0;
    double data_precision = 2.2204e-16; // MATLAB precision
    double old_delta = 0;
    int count_delta_stop = 0;
    
    double epsilon_old;
    double minEps;
    int new_lambda;
    int sgn_new_lambda;
    
    // Primal and Dual update variables
    int out_x, out_lambda;
    int outx_index, outl_index;
    double delta, theta;
    int i_delta, i_theta;
    
    // Primal and Dual increments
	Eigen::VectorXd dk, bk;
    
    // Auxiliar variables
    //Eigen::VectorXd del_x, del_lambda;   Both can be replaced by its globals
	Eigen::VectorXd pk_temp, ak_temp;
    Eigen::MatrixXd iAtgxAgl_ij, AtgxAgl_ij;
    Eigen::VectorXd AtgxAnl;
    Eigen::VectorXd Agdelx, Agdel_lam;
	Eigen::VectorXd temp_row, temp_col;            
    
    
    Eigen::MatrixXd AtglAgx = Eigen::MatrixXd::Zero(gamma_lambdak.size(), gamma_xk.size()); 
    Eigen::MatrixXd iAtglAgx = Eigen::MatrixXd::Zero(gamma_lambdak.size(), gamma_xk.size()); 
    Eigen::MatrixXd AtgxAgl = Eigen::MatrixXd::Zero(gamma_lambdak.size(), gamma_xk.size()); 
    Eigen::MatrixXd iAtgxAgl = Eigen::MatrixXd::Zero(gamma_lambdak.size(), gamma_xk.size()); 
    
    AtglAgx(0, 0) = A.col(gamma_lambdak[0]).dot(A.col(gamma_xk[0]));
    iAtglAgx(0, 0) = 1 / AtglAgx(0, 0);
    AtglAgx(0, 0) = AtglAgx(0, 0);
    iAtgxAgl(0, 0) = iAtglAgx(0, 0);
    
    if (verbose)
    {
        t1 = std::chrono::steady_clock::now();
        time = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();
        std::cout << "\nInitialization: " << time << std::endl;
    }
    
    while (iteration < maxIter)
    {
        if (verbose){t1 = std::chrono::steady_clock::now();}
        iteration++;
        
        // gamma_x = gamma_xk;
        gamma_x.clear(); 
        copy(gamma_xk.begin(), gamma_xk.end(), back_inserter(gamma_x)); 
        
        // gamma_lambda = gamma_lambdak;
        gamma_lambda.clear(); 
        copy(gamma_lambdak.begin(), gamma_lambdak.end(), back_inserter(gamma_lambda)); 
        
        // z_lambda = z_lambdak;
        z_lambda.resize(0); z_lambda = z_lambdak; 
        // z_x = z_xk;
        z_x.resize(0); z_x = z_xk; 
        // x_k = xk_1;
        x_k.resize(0); x_k = xk_1;
        // lambda_k = lambdak_1;
        lambda_k.resize(0); lambda_k = lambdak_1; 
        
        /////////////////
        // Update on x //
        /////////////////
        tempD.resize(0); tempD.setZero(gamma_lambda.size());
        del_x_vec.resize(0); del_x_vec.setZero(n);
                
        //Update direction
        for(size_t k = 0; k < gamma_lambda.size(); k++) {tempD(k) = z_lambda[gamma_lambda[k]];}
        
        for(size_t k = 0; k < gamma_x.size(); k++) {del_x_vec[gamma_x[k]] = -iAtglAgx.row(k).dot(tempD);}

        
        if (verbose)
        {
            t2 = std::chrono::steady_clock::now();
            time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        }
        
        // BOTTLENECK!!
        dk.resize(0);
        dk.setZero(n);
        
        Agdelx.resize(0);
        Agdelx.setZero(m);
        for(size_t k = 0; k < gamma_x.size(); k++) {Agdelx.noalias() += A.col(gamma_x[k]) * del_x_vec[gamma_x[k]];}
        dk = A.transpose().lazyProduct(Agdelx);
        
        ////// Precision control
        pk_temp.resize(0); pk_temp = Primal_constrk;
        minEps = epsilon < data_precision * 2 ? epsilon : data_precision * 2;
        
        for(int k = 0 ; k < n; k++)
        {
            if (fabs(fabs(Primal_constrk(k))-epsilon) < minEps)
                {pk_temp[k] = sign(Primal_constrk(k)) * epsilon;}
        }
        
        xk_temp = x_k;
        xk_temp.unaryExpr([&](double x) {return (fabs(x) < data_precision)? 0 : x;});
        
        update_primal(outx_index, delta, i_delta, pk_temp, dk);
        
        if ((old_delta < 4*data_precision) & (delta < 4*data_precision)) {count_delta_stop = count_delta_stop + 1;}
        else {count_delta_stop = 0;}
        
        if (count_delta_stop >= 50)
        {
            std::cerr << "\nStuck in some corner\n";
            break;
        }
        
        // Update variables
        old_delta = delta;
        xk_1 = x_k;
        xk_1.noalias() += delta*del_x_vec;
        Primal_constrk.noalias() += delta*dk;
        epsilon_old = epsilon;
        epsilon = epsilon - delta;     
        
       if (verbose) { std::cout << " --It: " << iteration << " Epsilon: " << epsilon;}
        
        if (epsilon < threshold)
        {
            //Primal_constrk.noalias() += (epsilon_old - threshold) * dk;
            xk_1 = x_k;
            xk_1.noalias() += + (epsilon_old - threshold) * del_x_vec;
            break;
        }
            //std::cout << std::endl << iteration << " (" << epsilon << ")";
        
        if (outx_index == -1)
        {
            // If a dual constraint becomes active, i_delta is its index
            if (verbose) {std::cout << "\tPrimal Add " << i_delta << "!";}
            gamma_lambdak.clear();
            copy(gamma_lambda.begin(), gamma_lambda.end(), back_inserter(gamma_lambdak)); 
            gamma_lambdak.push_back(i_delta);
            new_lambda = i_delta;
            lambda_k(new_lambda) = 0;
        }
        else
        {
            // If an element is removed from gamma_x, out_x is the element to remove (outx_index is its index)
            if (verbose) {std::cout << "\tPrimal Rem " << gamma_x[outx_index] << "!";}
            std::vector<int> gl_old(gamma_lambda);
            std::vector<int> gx_old(gamma_x);
            
            out_x = gamma_x[outx_index];
            gamma_x[outx_index] = gamma_x[gamma_x.size() - 1];
            gamma_x[gamma_x.size() - 1] = out_x;
            gamma_x.pop_back();// = gamma_x(1:end-1);
            
            iAtgxAgl.block(0, 0, gamma_lambdak.size(), gamma_xk.size()).col(outx_index).cwiseAbs().maxCoeff(&outl_index); 
            new_lambda = gamma_lambda[outl_index];
            
            gamma_lambda[outl_index] = gamma_lambda[gamma_lambda.size() - 1];
            gamma_lambda[gamma_lambda.size() - 1] = new_lambda;
            
            gamma_lambdak.clear();
            copy(gamma_lambda.begin(), gamma_lambda.end(), back_inserter(gamma_lambdak)); 
            gamma_lambda.pop_back();
            
            // outx_index: ith row of A is swapped with last row (out_x)
            // outl_index: jth column of A is swapped with last column (out_lambda)
            
            AtgxAgl_ij = AtgxAgl.eval();
            temp_row = AtgxAgl_ij.row(outx_index);
            temp_col = AtgxAgl_ij.col(outl_index);
            AtgxAgl_ij.block(outx_index, 0, AtgxAgl_ij.rows() - outx_index, AtgxAgl_ij.cols()).noalias() = AtgxAgl_ij.block(outx_index + 1,0,AtgxAgl_ij.rows() - outx_index,AtgxAgl_ij.cols());
            AtgxAgl_ij.block(0, outl_index, AtgxAgl_ij.rows(), AtgxAgl_ij.cols() - outl_index).noalias() = AtgxAgl_ij.block(0,outl_index + 1,AtgxAgl_ij.rows(),AtgxAgl_ij.cols() - outl_index);
            AtgxAgl_ij.row(AtgxAgl_ij.rows() - 1) = temp_row;
            AtgxAgl_ij.col(AtgxAgl_ij.cols() - 1) = temp_col;
            
            // Lost rows and columns needed for the inversion
            iAtgxAgl_ij = iAtgxAgl.eval();
            temp_row = iAtgxAgl_ij.row(outl_index);
            temp_col = iAtgxAgl_ij.col(outx_index);
            iAtgxAgl_ij.block(outl_index, 0, iAtgxAgl_ij.rows() - outl_index, iAtgxAgl_ij.cols()).noalias() = iAtgxAgl_ij.block(outl_index + 1,0, iAtgxAgl_ij.rows() - outl_index, iAtgxAgl_ij.cols());
            iAtgxAgl_ij.block(0, outx_index, iAtgxAgl_ij.rows(), iAtgxAgl_ij.cols() - outx_index).noalias() = iAtgxAgl_ij.block(0,outx_index + 1, iAtgxAgl_ij.rows(), iAtgxAgl_ij.cols() - outx_index);
            iAtgxAgl_ij.row(iAtgxAgl_ij.rows() - 1) = temp_row;
            iAtgxAgl_ij.col(iAtgxAgl_ij.cols() - 1) = temp_col;
            
            AtgxAgl = AtgxAgl_ij.block(0, 0, gamma_lambda.size(), gamma_x.size());
            AtglAgx = AtgxAgl.transpose();
            update_inverse(AtgxAgl_ij, iAtgxAgl_ij, iAtgxAgl, 2);

            iAtglAgx = iAtgxAgl.transpose();
            xk_1(outx_index) = 0;
        }
        //out_lambda.clear();
        z_lambdak.resize(0);
        z_lambdak.setZero(n);// = Eigen::VectorXi::Zero(n);
        for (auto const &g_lk: gamma_lambdak)
        { 
            z_lambdak(g_lk) = sign(Primal_constrk(g_lk));
            Primal_constrk(g_lk) = sign(Primal_constrk(g_lk))*epsilon;
        }
        sgn_new_lambda = sign(Primal_constrk(new_lambda));
        
        if (verbose)
        {
            t2 = std::chrono::steady_clock::now();
            std::cout << " (" << std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count() - time << " s) ";
            time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        }
        
        //////////////////////
        // Update on lambda //
        //////////////////////
        
        AtgxAnl.resize(0);
        AtgxAnl.setZero(gamma_x.size());
        for(size_t k = 0; k < gamma_x.size(); k++) {AtgxAnl(k) = A.col(gamma_x[k]).transpose() * A.col(new_lambda);}

        // Update direction
        //del_lambda.resize(0);
        //del_lambda = -iAtgxAgl*AtgxAnl;
        del_lambda_p.resize(0);
        del_lambda_p.setZero(n);
        for (size_t k = 0; k < gamma_lambda.size(); k++)
            {del_lambda_p(gamma_lambda[k]) = -iAtgxAgl.row(k).dot(AtgxAnl) * sgn_new_lambda;}
        del_lambda_p(new_lambda) = 1*sgn_new_lambda;
        
        //ak = Dual_constrk;
        gamma_lambda.push_back(new_lambda); 
        bk.resize(0);
        bk.setZero(n);// = Eigen::VectorXd::Zero(n);
        
        Agdel_lam.resize(0);
        Agdel_lam.setZero(m);
        for(size_t k = 0; k < gamma_lambda.size(); k++) {Agdel_lam.noalias() += A.col(gamma_lambda[k]) * del_lambda_p(gamma_lambda[k]);}
        bk = A.transpose().lazyProduct(Agdel_lam);
        
        gamma_lambda.pop_back(); 
        
        // check if the sign of update direction is correct
        if ((outx_index != -1) && (sign(bk(out_x)) == sign(Dual_constrk(out_x))) & (abs(bk(out_x)) >= data_precision))
        {
            bk.noalias() = -bk;
            del_lambda_p.noalias() = -del_lambda_p;
        }
        
        // CONTROL THE MACHINE PRECISION ERROR AT EVERY OPERATION: LIKE BELOW. 
        ak_temp.resize(0);
        ak_temp = Dual_constrk;
        
        for(int k = 0 ; k < n; k++)
        {
            if (fabs(fabs(Dual_constrk(k))-1) < 2 * data_precision)
                ak_temp[k] = sign(Primal_constrk(k));
        }      
        
        lambdak_temp = lambda_k;
        lambdak_temp.unaryExpr([&](double x) {return (fabs(x) < data_precision)? 0 : x;});
        
        update_dual(outl_index, theta, i_theta, ak_temp, bk, new_lambda);
        
        lambdak_1.noalias() = lambda_k + theta*del_lambda_p;
        Dual_constrk.noalias() += theta * bk;
        
        if (outl_index == -1)
        {
            if (verbose) {std::cout << "\tDual Add " << i_theta << "!";}
            //If an element is added to gamma_x
            Eigen::MatrixXd AtglAgx_mod = Eigen::MatrixXd::Zero(gamma_lambda.size() + 1, gamma_x.size() + 1);
            AtglAgx_mod.block(0, 0, gamma_lambdak.size(), gamma_xk.size()) = AtglAgx; // AtglAgx_mod 11
            AtglAgx_mod.row(gamma_lambda.size()) = AtgxAnl.transpose(); // AtglAgx_mod 21
            for (size_t k = 0; k < gamma_lambda.size(); k++)
                {AtglAgx_mod(k, gamma_x.size()) = A.col(gamma_lambda[k]).transpose() * A.col(i_theta);}// AtglAgx_mod 12
            // AtglAgx_mod 22
            AtglAgx_mod(gamma_lambda.size(), gamma_x.size()) = A.col(new_lambda).transpose() * A.col(i_theta); 
            // CHECK THIS SINGULARITY CONDITION USING SCHUR COMPLEMENT IDEA !!!
            //  X = [A B; C D];
            //  detX = detA detS
            //  S = D-C A^{-1} B
            // A11 = AtglAgx;
            // A12 = AtglAgx_mod(1:end-1,end);
            // A21 = AtglAgx_mod(end,1:end-1);
            // A22 = AtglAgx_mod(end,end);
            // S = A22 - A21*(iAtglAgx*A12);
            
            if (abs(AtglAgx_mod(gamma_lambda.size(), gamma_x.size()) - AtglAgx_mod.row(gamma_lambda.size()).dot(iAtglAgx * AtglAgx_mod.col(gamma_x.size()))) < 2*data_precision)
            {
                std::cerr << "\n\tMatrix has become singular\n";
                break;
            }
            AtglAgx = AtglAgx_mod;
            AtgxAgl = AtglAgx.transpose();
            Eigen::MatrixXd iAtglAgx_old = iAtglAgx;
            update_inverse(AtglAgx, iAtglAgx_old, iAtglAgx, 1);
            iAtgxAgl = iAtglAgx.transpose();

            out_lambda = 0;
            gamma_lambda.push_back(new_lambda);
            gamma_x.push_back(i_delta);
            
            z_xk.setZero(n);
            for (auto const &g_x: gamma_x)
                {z_xk(g_x) = -sign(Dual_constrk(g_x));}
            xk_1(i_theta) = 0;
        }
        else
        {
            if (verbose) {std::cout << "\tDual Rem " << gamma_lambda[outl_index] << "!";}
            out_lambda = gamma_lambda[outl_index];
            i_theta = gamma_x.back();
            
            Eigen::MatrixXd iA = iAtgxAgl;
            //Eigen::VectorXd C = Eigen::VectorXd::Zero(gamma_lambda.size());
            //C(outl_index) = 1;
            AtgxAgl.col(outl_index) = AtgxAnl;
            AtglAgx = AtgxAgl.transpose();

            Eigen::VectorXd B = AtgxAnl;
            for(size_t k = 0; k < gamma_x.size(); k++) 
                {B.row(k).noalias() -= A.col(gamma_x[k]).transpose() * A.col(out_lambda);}
            Eigen::MatrixXd iAB = iA*B;
            iAtgxAgl = iA-iAB*((iA.row(outl_index))/(1+iAB(outl_index)));
            iAtglAgx = iAtgxAgl.transpose();
            
            gamma_lambda[outl_index] = new_lambda;
            z_xk.resize(0);
            z_xk.setZero(n);
            for (auto const &g_x: gamma_x)
                {z_xk(g_x) = -sign(Dual_constrk(g_x));}
            lambdak_1(out_lambda) = 0;
        }
        for (auto const &g_x: gamma_x)
            {Dual_constrk(g_x) = sign(Dual_constrk(g_x));}
            
        gamma_lambdak = gamma_lambda;
        gamma_xk = gamma_x;
        
        if (verbose)
        {
            t2 = std::chrono::steady_clock::now();
            std::cout << " (" << std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count() - time << " s) ";
        } 
        
        if (verbose)
        {
            t2 = std::chrono::steady_clock::now();
            time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            std::cout << "\t It: " << time << " s" << std::endl;
        }
    }
    
    if (verbose)
    {
        t2 = std::chrono::steady_clock::now();
        time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t0).count();
        std::cout << "\n\t Total: " << time << " s" << std::endl;
    }
}

void  DSHomotopy::update_primal(int &out_xi, double &delta, int &i_delta, const Eigen::VectorXd &pk, const Eigen::VectorXd &dk)//, double epsilon, std::vector<double> out_lambda)
{
    std::vector<int> gamma_lc(x_k.size());
	std::iota(gamma_lc.begin(), gamma_lc.end(), 0);
	// Must be sure that the value is inside!
	for (auto const &g_lam: gamma_lambda)
		{gamma_lc.erase(std::find(gamma_lc.begin(), gamma_lc.end(), g_lam));}

    int delta_pos_len;
    int min_index;
    double temp;
    
    Eigen::VectorXd delta_pos = Eigen::VectorXd::Zero(gamma_lc.size()); 
    
    /////////////
    // DELTA 1 //
    /////////////
    delta_pos_len = 0;
    min_index = 0;
    double delta1;
    int i_delta1 = -1;
    Eigen::VectorXi delta1_pos_ind = Eigen::VectorXi::Zero(gamma_lc.size()); 
    
    for (size_t k=0; k < gamma_lc.size(); k++)
    {
        temp = (epsilon - pk(gamma_lc[k]))/(1 + dk(gamma_lc[k]));
        if ( temp > 0 )
        {
            delta1_pos_ind(delta_pos_len) = k;
            delta_pos(delta_pos_len) = temp;
            delta_pos_len++;
        }
    }
    
    if (delta_pos_len == 0) 
    {
        delta1 = INT_MAX;
        i_delta1 = -1;        
    }
    else
    {
        delta1 = delta_pos.head(delta_pos_len - 1).array().minCoeff(&min_index); 
        i_delta1 = min_index;
    }
    
    /////////////
    // DELTA 2 //
    /////////////
    delta_pos.setZero();
    delta_pos_len = 0;
    min_index = 0;
    double delta2;
    int i_delta2;
    Eigen::VectorXi delta2_pos_ind = Eigen::VectorXi::Zero(gamma_lc.size()); 

    for (size_t k=0; k < gamma_lc.size(); k++)
    {
        temp = (epsilon + pk(gamma_lc[k]))/(1 - dk(gamma_lc[k]));
        if ( temp > 0 )
        {
            delta2_pos_ind(delta_pos_len) = k;
            delta_pos(delta_pos_len) = temp;
            delta_pos_len++;
        }
    }

    if (delta_pos_len == 0) 
    {
        delta2 = INT_MAX;
        i_delta2 = -1;
    }
    else
    {
        delta2 = delta_pos.head(delta_pos_len - 1).array().minCoeff(&min_index); 
        i_delta2 = min_index;
    }
    
    /////////////
    // COMP
    /////////////
    if (delta1 > delta2){
        delta   = delta2;
        i_delta = delta2_pos_ind[i_delta2];
    }else{
        delta   = delta1;
        i_delta = delta1_pos_ind[i_delta1];
    }
    
    /////////////
    // DELTA 3 //
    /////////////
    delta_pos.setZero();
    delta_pos_len = 0;
    min_index = 0;
    double delta3;
    int i_delta3;
    
    Eigen::VectorXi delta3_pos_ind = Eigen::VectorXi::Zero(gamma_lc.size());
    for (size_t k=0; k < gamma_x.size(); k++)
    {
        temp = -1 * xk_temp(gamma_x[k]) / del_x_vec(gamma_x[k]);
        if ( temp > 0 )
        {
            delta3_pos_ind(delta_pos_len) = k;
            delta_pos(delta_pos_len) = temp;
            delta_pos_len++;
        }
    }

    if (delta_pos_len == 0) 
    {
        delta3 = INT_MAX;
        i_delta3 = -1;
    }else{
        delta3 = delta_pos.head(delta_pos_len - 1).array().minCoeff(&min_index); 
        i_delta3 = min_index;
    }

    /////////////
    // FINAL
    /////////////
    out_xi = -1;
    if (delta3 > 0 && delta3 <= delta)
    {
        delta = delta3;
        out_xi = delta3_pos_ind(i_delta3);
    }
    i_delta = gamma_lc.at(i_delta);
}

void  DSHomotopy::update_dual(int &out_lambdai, double &theta, int &i_theta, const Eigen::VectorXd &ak, const Eigen::VectorXd &bk, int new_lambda)//, std::vector<double> out_lambda)
{
    std::vector<int> gamma_xc(lambda_k.size());
	std::iota(gamma_xc.begin(), gamma_xc.end(), 0);
	// Must be sure that the value is inside!
	for (auto const &g_lam: gamma_lambda)
		{gamma_xc.erase(std::find(gamma_xc.begin(), gamma_xc.end(), g_lam));}

    int theta_pos_len;
    int min_index;
    double temp;
    
    Eigen::VectorXd theta_pos = Eigen::VectorXd::Zero(gamma_xc.size()); 
    
    /////////////
    // THETA 1 //
    /////////////
    theta_pos_len = 0;
    min_index = 0;
    double theta1;
    int i_theta1 = -1;
    Eigen::VectorXi theta1_pos_ind = Eigen::VectorXi::Zero(gamma_xc.size()); 
    
    for (size_t k=0; k < gamma_xc.size(); k++)
    {
        temp = (1 - ak(gamma_xc[k]))/(bk(gamma_xc[k]));
        if ( temp > 0 )
        {
            theta1_pos_ind(theta_pos_len) = k;
            theta_pos(theta_pos_len) = temp;
            theta_pos_len++;
        }
    }
    
    if (theta_pos_len == 0) 
    {
        theta1 = INT_MAX;
        i_theta1 = -1;        
    }
    else
    {
        theta1 = theta_pos.head(theta_pos_len - 1).array().minCoeff(&min_index); 
        i_theta1 = min_index;
    }
    
    /////////////
    // THETA 2 //
    /////////////
    theta_pos.setZero();
    theta_pos_len = 0;
    min_index = 0;
    double theta2;
    int i_theta2;
    Eigen::VectorXi theta2_pos_ind = Eigen::VectorXi::Zero(gamma_xc.size()); 

    for (size_t k=0; k < gamma_xc.size(); k++)
    {
        temp = - (1 + ak(gamma_xc[k]))/(bk(gamma_xc[k]));
        if ( temp > 0 )
        {
            theta2_pos_ind(theta_pos_len) = k;
            theta_pos(theta_pos_len) = temp;
            theta_pos_len++;
        }
    }

    if (theta_pos_len == 0) 
    {
        theta2 = INT_MAX;
        i_theta2 = -1;
    }
    else
    {
        theta2 = theta_pos.head(theta_pos_len - 1).array().minCoeff(&min_index); 
        i_theta2 = min_index;
    }
    
    /////////////
    // COMP
    /////////////
    if (theta1 > theta2){
        theta   = theta2;
        i_theta = theta2_pos_ind[i_theta2];
    }else{
        theta   = theta1;
        i_theta = theta1_pos_ind[i_theta1];
    }
    
    /////////////
    // THETA 3 //
    /////////////
    theta_pos.setZero();
    theta_pos_len = 0;
    min_index = 0;
    double theta3;
    int i_theta3;
    
    Eigen::VectorXi theta3_pos_ind = Eigen::VectorXi::Zero(gamma_xc.size());
    gamma_lambda.push_back(new_lambda);
    
    for (size_t k=0; k < gamma_lambda.size(); k++)
    {
        temp = -1 * lambda_k(gamma_lambda[k]) / del_lambda_p(gamma_lambda[k]);
        if ( temp > 0 )
        {
            theta3_pos_ind(theta_pos_len) = k;
            theta_pos(theta_pos_len) = temp;
            theta_pos_len++;
        }
    }

    if (theta_pos_len == 0) 
    {
        theta3 = INT_MAX;
        i_theta3 = -1;
    }else{
        theta3 = theta_pos.head(theta_pos_len - 1).array().minCoeff(&min_index); 
        i_theta3 = min_index;
    }

    /////////////
    // FINAL
    /////////////
    out_lambdai = -1;
    if (theta3 > 0 && theta3 <= theta)
    {
        theta = theta3;
        out_lambdai = theta3_pos_ind(i_theta3);
    }
    
    i_theta = gamma_xc.at(i_theta);
    gamma_lambda.pop_back();
}

void DSHomotopy::update_inverse( const Eigen::MatrixXd &AtB, const Eigen::MatrixXd &iAtB_old, Eigen::MatrixXd &iAtB, int flag)
{
    //A12 = AtB(1:n-1,n); col(n)
    //A21 = AtB(n,1:n-1); row(n)
    //A22 = AtB(n,n);
    
    int siz = AtB.rows();

    if (flag == 1)
    {
        // Add columns
        Eigen::VectorXd iA11A12 = iAtB_old * AtB.col(siz - 1);
        Eigen::VectorXd A21iA11 =  AtB.row(siz - 1) * iAtB_old;
        double S = AtB(siz - 1, siz - 1) - (AtB.row(siz - 1).dot(iA11A12)); // 
        Eigen::MatrixXd Q11_right = iA11A12 * (A21iA11.transpose()/S); 

        iAtB.setZero(siz, siz);
        iAtB.block(0, 0, siz-1, siz-1) = iAtB_old+ Q11_right;
        iAtB.col(siz - 1) = -iA11A12 / S; 
        iAtB.row(siz - 1) =  -A21iA11 / S;
        iAtB(siz - 1, siz - 1) = 1 / S;
    }
    else if (flag == 2)
    {
        // Delete columns
        Eigen::MatrixXd Q12Q21_Q22 = iAtB_old.col(siz - 1).head(siz-1) * (iAtB_old.row(siz - 1).head(siz-1) / iAtB_old(siz-1, siz-1));
        iAtB.resize(0, 0);
        iAtB = iAtB_old.block(0, 0, siz - 1, siz - 1) - Q12Q21_Q22;
    }
    else
    {
        std::cerr << std::endl << "Not observed mode for the inverse update!!" << std::endl;
        exit(-1);
    }
}
