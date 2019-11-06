#include "BPDN_homotopy.h"

#include <iostream>
#include <iomanip>
#include <numeric>
#include <chrono>


void BPDNHomotopy::solveHomotopy(const Eigen::VectorXd &y, const Eigen::MatrixXd &A, Eigen::VectorXd& xk_1)
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
    gamma_x.clear(); 
    
    // Initial step
    Eigen::VectorXd Primal_constrk = -A.transpose() * y;    
    int i; 
    double c = Primal_constrk.array().abs().maxCoeff(&i); 
    
    std::vector<int> gamma_xk = {i};
    if (verbose) {std::cout << "\n\tPrimal Add " << i << "!" << "\tDual Add " << i << "!";}
    
    epsilon = c;
    xk_1.setZero(n); // Final primal Solution    
    x_k.resize(0); x_k.setZero(n);
    
    z_x(gamma_xk[0]) = -sign(Primal_constrk(gamma_xk[0]));
    Primal_constrk(gamma_xk[0]) = sign(Primal_constrk(gamma_xk[0])) * epsilon;
    
    Eigen::VectorXi z_xk(z_x);
    
    // Loop variables
    int iteration = 0;
    double data_precision = 2.2204e-16; // MATLAB precision
    double old_delta = 0;
    int count_delta_stop = 0;
    
    double epsilon_old;
    double minEps;
    int new_x;
    
    // Primal and Dual update variables
    int out_x;
    int outx_index;
    double delta;
    int i_delta;
    
    // Primal and Dual increments
	Eigen::VectorXd dk;
    
    // Auxiliar variables
    //Eigen::VectorXd del_x, del_lambda;   Both can be replaced by its globals
	Eigen::VectorXd pk_temp;
    Eigen::MatrixXd AtAgx_ij, iAtAgx_ij;
    Eigen::VectorXd AtgxAnx;
    Eigen::VectorXd Agdelx;
	Eigen::VectorXd temp_row, temp_col;            
    
    
    Eigen::MatrixXd AtAgx = Eigen::MatrixXd::Zero(gamma_xk.size(), gamma_xk.size()); 
    Eigen::MatrixXd iAtAgx = Eigen::MatrixXd::Zero(gamma_xk.size(), gamma_xk.size()); 
    
    AtAgx(0, 0) = A.col(gamma_xk[0]).dot(A.col(gamma_xk[0]));
    iAtAgx(0, 0) = 1 / AtAgx(0, 0);
    
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
        
        z_x = z_xk; 

        x_k.resize(0); x_k = xk_1;
        
        /////////////////
        // Update on x //
        /////////////////
        tempD.resize(0); tempD.setZero(gamma_x.size());
        del_x_vec.resize(0); del_x_vec.setZero(n);
                
        //Update direction
        for(size_t k = 0; k < gamma_x.size(); k++) {tempD(k) = z_x[gamma_x[k]];}
        
        for(size_t k = 0; k < gamma_x.size(); k++) {del_x_vec[gamma_x[k]] = -iAtAgx.row(k).dot(tempD);}

        
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
        
        // Stop breaking
        if ((epsilon - delta) < 0 || (static_cast<int>(gamma_x.size() - 1) == n))
        {
            std::cerr << std::endl << "Stopped!";
            break;
        }
        
        // Update variables
        old_delta = delta;
        xk_1 = x_k;
        xk_1.noalias() += delta * del_x_vec;
        Primal_constrk.noalias() += delta * dk;
        epsilon_old = epsilon;
        epsilon = epsilon - delta;     
        
       if (verbose) { std::cout << " --It: " << iteration << " Epsilon: " << epsilon;}             
        
        if (epsilon < threshold)
        {
            //Primal_constrk.noalias() += (epsilon_old - threshold) * dk;
            xk_1 = x_k;
            xk_1.noalias() += (epsilon_old - threshold) * del_x_vec;
            break;
        }
            //std::cout << std::endl << iteration << " (" << epsilon << ")";
        
        if (outx_index == -1)
        {
            // If a dual constraint becomes active, i_delta is its index
            if (verbose) {std::cout << "\tPrimal Add " << i_delta << "!";}            
            // Here we force that the support of primal and dual vectors remain same at every step
            gamma_xk.clear();
            copy(gamma_x.begin(), gamma_x.end(), back_inserter(gamma_xk)); 
            gamma_xk.push_back(i_delta);
            new_x = i_delta;
            
            // Update AtgxAnl = A(:,gamma_x)'*A(:,new_lambda);
            AtgxAnx.resize(0);
            AtgxAnx.setZero(gamma_x.size());
            for(size_t k = 0; k < gamma_x.size(); k++) {AtgxAnx(k) = A.col(gamma_x[k]).transpose() * A.col(new_x);}
            
            // Update AtAgx_mod = [AtAgx AtgxAnx; AtgxAnx' A(:,new_x)'*A(:,idelta)];
            Eigen::MatrixXd AtAgx_mod = Eigen::MatrixXd::Zero(gamma_x.size() + 1, gamma_x.size() + 1);
            AtAgx_mod.block(0, 0, gamma_x.size(), gamma_xk.size()) = AtgxAnx; // AtAgx_mod 11
            AtAgx_mod.row(gamma_x.size()) = AtgxAnx.transpose(); // AtAgx_mod 21
            for (size_t k = 0; k < gamma_x.size(); k++)
                {AtAgx_mod(k, gamma_x.size()) = A.col(gamma_x[k]).transpose() * A.col(i_delta);}// AtAgx_mod 12
            // AtAgx_mod 22
            AtAgx_mod(gamma_x.size(), gamma_x.size()) = A.col(new_x).transpose() * A.col(i_delta); 
            
            AtAgx = AtAgx_mod;
            Eigen::MatrixXd iAtglAgx_old = iAtAgx;
            update_inverse(AtAgx, iAtglAgx_old, iAtAgx, 1);
            xk_1(i_delta) = 0;
            
            gamma_x.clear();
            copy(gamma_xk.begin(), gamma_xk.end(), back_inserter(gamma_x)); 
        }
        else
        {
            // If an element is removed from gamma_x, out_x is the element to remove (outx_index is its index)
            if (verbose) {std::cout << "\tPrimal Rem " << gamma_x[outx_index] << "!";}
            
            out_x = gamma_x[outx_index];
            gamma_x[outx_index] = gamma_x[gamma_x.size() - 1];
            gamma_x[gamma_x.size() - 1] = out_x;
            gamma_x.pop_back();// = gamma_x(1:end-1);
            
            gamma_xk.clear();
            copy(gamma_x.begin(), gamma_x.end(), back_inserter(gamma_xk)); 
            
            // outx_index: ith row of A is swapped with last row (out_x)
            // outl_index: jth column of A is swapped with last column (out_lambda)

            AtAgx_ij = AtAgx.eval();
            temp_row = AtAgx_ij.row(outx_index);
            temp_col = AtAgx_ij.col(outx_index);
            AtAgx_ij.block(outx_index, 0, AtAgx_ij.rows() - outx_index, AtAgx_ij.cols()).noalias() = AtAgx_ij.block(outx_index + 1,0,AtAgx_ij.rows() - outx_index,AtAgx_ij.cols());
            AtAgx_ij.block(0, outx_index, AtAgx_ij.rows(), AtAgx_ij.cols() - outx_index).noalias() = AtAgx_ij.block(0,outx_index + 1,AtAgx_ij.rows(),AtAgx_ij.cols() - outx_index);
            AtAgx_ij.row(AtAgx_ij.rows() - 1) = temp_row;
            AtAgx_ij.col(AtAgx_ij.cols() - 1) = temp_col;
            
            // Lost rows and columns needed for the inversion
            iAtAgx_ij = iAtAgx.eval();
            temp_row = iAtAgx_ij.row(outx_index);
            temp_col = iAtAgx_ij.col(outx_index);
            iAtAgx_ij.block(outx_index, 0, iAtAgx_ij.rows() - outx_index, iAtAgx_ij.cols()).noalias() = iAtAgx_ij.block(outx_index + 1,0, iAtAgx_ij.rows() - outx_index, iAtAgx_ij.cols());
            iAtAgx_ij.block(0, outx_index, iAtAgx_ij.rows(), iAtAgx_ij.cols() - outx_index).noalias() = iAtAgx_ij.block(0,outx_index + 1, iAtAgx_ij.rows(), iAtAgx_ij.cols() - outx_index);
            iAtAgx_ij.row(iAtAgx_ij.rows() - 1) = temp_row;
            iAtAgx_ij.col(iAtAgx_ij.cols() - 1) = temp_col;
            
            AtAgx = AtAgx_ij.block(0, 0, gamma_x.size(), gamma_x.size());
            update_inverse(AtAgx_ij, iAtAgx_ij, iAtAgx, 2);
            
            xk_1(outx_index) = 0;
        }
        
        z_xk.resize(0); z_xk.setZero(n);
        for(auto const &g_xk: gamma_xk) 
        {
            z_xk(g_xk) = -sign(Primal_constrk(g_xk));
            Primal_constrk(g_xk) = sign(Primal_constrk(g_xk)) * epsilon;
            
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


void  BPDNHomotopy::update_primal(int &out_xi, double &delta, int &i_delta, const Eigen::VectorXd &pk, const Eigen::VectorXd &dk)//, double epsilon, std::vector<double> out_lambda)
{
    std::vector<int> gamma_lc(x_k.size());
	std::iota(gamma_lc.begin(), gamma_lc.end(), 0);
	// Must be sure that the value is inside!
	for (auto const &g_x: gamma_x)
		{gamma_lc.erase(std::find(gamma_lc.begin(), gamma_lc.end(), g_x));}

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

void  BPDNHomotopy::update_inverse( const Eigen::MatrixXd &AtB, const Eigen::MatrixXd &iAtB_old, Eigen::MatrixXd &iAtB, int flag)
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
