/***************************************************************************************

Copyright (October 26 2015)

Authors:
  Gaétan Marceau Caron (INRIA-Saclay)
  gaetan.marceau-caron@inria.fr

  Yann Olliver (CNRS & Paris-Saclay University)
  yann.ollivier@lri.fr

  This work has been partially funded by the French cooperative project TIMCO, Pôle de Compétitivité Systematic (FUI 13).

  This software is a computer program whose purpose is to provide an experimental framework
  for research in Deep Learning and Riemannian optimization. 
  
  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".
  
  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.
  
  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL license and that you accept its terms.
**************************************************************************************/

/*!
 * \file nn_ops_sp_conv.hpp
 * \brief Implementation of the functions required for neural networks 
 * \author Gaetan Marceau Caron & Yann Ollivier
 * \version 1.0
 */

#include <functional>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <algorithm>
#include <assert.h>
#include <limits> 
#include <algorithm>

#include "utils.hpp"

void initSlidingMetric(const std::vector<MyMatrix> &W,
		       std::vector<MyMatrix> &pMii,
		       std::vector<MyMatrix> &pM0i,
		       std::vector<MyVector> &pM00){
    
  pMii.resize(W.size());
  pM0i.resize(W.size());
  pM00.resize(W.size());

  for(unsigned i = 0; i < pMii.size(); i++){    
    pMii[i] = MyMatrix::Ones(W[i].rows(), W[i].cols());
    pM0i[i] = MyMatrix::Zero(W[i].rows(), W[i].cols());
    pM00[i] = MyVector::Ones(W[i].cols());
  }
}

void transposeConvW(const MyMatrix& conv_W,
		    const unsigned n_chan,
		    const unsigned Hf,
		    MyMatrix &conv_W_T){


  const unsigned n_filter = conv_W.rows();
  conv_W_T.setZero(n_chan, Hf*Hf*n_filter);
  for(unsigned i = 0; i < n_filter; i++){
    for(unsigned j = 0; j < conv_W.cols(); j++){
      const unsigned idx1 = j/(Hf*Hf);
      const unsigned idx2 = (Hf*Hf-j%(Hf*Hf)-1)+i*Hf*Hf;
      conv_W_T(idx1,idx2) = conv_W(i,j);
    }
  }
}

void initFullLayer(const unsigned n_act0,
		   const unsigned n_act1,
		   const double sigma,
		   MyMatrix& W){
  
  normal_distribution<double> normal(0,sigma/sqrt(n_act0));
  W.resize(n_act0,n_act1);
  for(unsigned i = 0; i < n_act1; i++){
    for(unsigned j = 0; j < n_act0; j++){
      W(j,i) = normal(gen);
    }
  }
}

void initSparseLayer(const unsigned sparsity,
		     const unsigned n_act0,
		     const unsigned n_act1,
		     const double sigma,
		     unsigned &param_counter,
		     std::vector<SpEntry> &coeff){

  assert(sparsity < n_act0);

  std::uniform_real_distribution<double> uniform(0.0,1.0);

  // Generate randomly the sparse network
  for(unsigned i = 0; i < n_act1; i++){
    double u = 0.;
    unsigned t = 0; // total input records dealt with
    unsigned m = 0; // number of items selected so far

    // Knuth's Algorithm of sampling without replacement
    while (m < sparsity){
      u = uniform(gen); // call a uniform(0,1) random number generator
      if((n_act0 - t)*u >= sparsity - m ){
	t++;
      }
      else{
	coeff[i*sparsity+m] = SpEntry(t,i,1.);
	t++; m++;
      }
    }
  }

  std::cout << "coeff size " << n_act0 << " " << n_act1 << " " << coeff.size() << std::endl;

  // Generate randomly the weights
  for(unsigned i = 0; i < coeff.size(); i++){
    auto temp = coeff[i];
    normal_distribution<double> normal(0,sigma/sqrt(sparsity));
    const double weight = normal(gen);
    coeff[i] = SpEntry(temp.row(), temp.col(), weight);
    param_counter++;
  }
}


unsigned initConvLayer(const std::vector<ConvLayerParams> &conv_params,
		   std::vector<MyMatrix> &convW,
		   std::vector<MyMatrix> &convW_T,
		   std::vector<MyVector> &convB){

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,0.01);

  unsigned n_params = 0;
  for(unsigned l = 0; l < conv_params.size(); l++){
    const unsigned n_filter = conv_params[l].n_filter;
    const unsigned Hf = conv_params[l].Hf;
    unsigned n_channel = N_CHANNEL;
    if(l>0){
      n_channel = conv_params[l-1].n_filter;
    }
    convW[l].resize(n_filter, Hf*Hf*n_channel);
    convB[l].resize(n_filter);
    for(unsigned i = 0; i < n_filter; i++){
      convB[l](i) = distribution(generator);
      for(unsigned j = 0; j < n_channel; j++){
	for(unsigned ko = 0; ko < Hf; ko++){
	  for(unsigned ki = 0; ki < Hf; ki++){
	    convW[l](i,j*Hf*Hf+ko*Hf+ki) = distribution(generator);
	    n_params++;
	  }
	}
      }
    }
    transposeConvW(convW[l],n_channel, Hf,convW_T[l]);
  }
}

// void initConvLayer(const unsigned n_filter,
// 		   const unsigned F,
// 		   MyMatrix &filters,
// 		   MyVector &B){

//   std::default_random_engine generator;
//   std::normal_distribution<double> distribution(0.0,0.01);
  
//   filters.resize(n_filter, F*F*N_CHANNEL);
//   for(unsigned i = 0; i < n_filter; i++){
//     B[i] = distribution(generator);
//     for(unsigned j = 0; j < N_CHANNEL; j++){
//       for(unsigned ko = 0; ko < F; ko++){
// 	for(unsigned ki = 0; ki < F; ki++){
// 	  filters(i,j*F*F+ko*F+ki) = distribution(generator);
// 	}
//       }
//     }
//   }
// }


int initNetwork(const std::vector<unsigned> &nn_arch, 
		const std::string act_func, 
		const unsigned sparsity,
		const std::vector<ConvLayerParams> &conv_params,
		const std::vector<PoolLayerParams> &pool_params,
		MyMatrix &W_out,
		std::vector<MySpMatrix> &W,
		std::vector<MySpMatrix> &Wt,
		std::vector<MyVector> &B,
		std::vector<MyMatrix> &convW,
		std::vector<MyMatrix> &convW_T,
		std::vector<MyVector> &convB){
  
  // Fix some parameters
  const unsigned n_layers = nn_arch.size();
  double sigma = 1.0;
  if(act_func=="sigmoid"){
    sigma = 4.0;
  }
  unsigned param_counter = 0;

  // Initialize the convolutional layer
  param_counter += initConvLayer(conv_params, convW, convW_T, convB);

  // Initialize the sparse weights
  for(unsigned i = 0; i < n_layers-2; i++){

    std::vector<SpEntry> coeff(nn_arch[i+1]*sparsity);
    initSparseLayer(sparsity, nn_arch[i], nn_arch[i+1], sigma, param_counter, coeff);
    param_counter += nn_arch[i+1];

    MySpMatrix spW(nn_arch[i], nn_arch[i+1]);
    spW.setFromTriplets(coeff.begin(), coeff.end());
    W[i] = spW;
    MySpMatrix spWt = spW.transpose();
    Wt[i] = spWt;

    if(act_func=="sigmoid"){
      MyMatrix temp = spW;
      MyVector b = -0.5 * temp.colwise().sum();
      B[i] = b;
    }else if(act_func=="tanh"){
      MyVector b = MyVector::Zero(W[i].cols());
      B[i] = b;
    }else if(act_func=="relu"){ // TODO: find the right way to initialize relu
      MyVector b = MyVector::Zero(W[i].cols());
      B[i] = b;
    }else{
      std::cout << "Not implemented!" << std::endl;
      assert(false);
    }
  }

  // Initialize the output weights
  initFullLayer(nn_arch[n_layers-2], nn_arch[n_layers-1], sigma, W_out);
  if(act_func=="sigmoid"){
    B[n_layers-2] = -0.5 * W_out.colwise().sum();
  }else if(act_func=="tanh"){
    B[n_layers-2] = MyVector::Zero(W_out.cols());
  }else if(act_func=="relu"){ // TODO: find the right way to initialize relu
    B[n_layers-2] = MyVector::Zero(W_out.cols());
  }else{
    std::cout << "Not implemented!" << std::endl;
    assert(false);
  }

  param_counter += nn_arch[n_layers-2] * nn_arch[n_layers-1] + nn_arch[n_layers-1];
  return param_counter;
  
}

// int initNetwork(const std::vector<unsigned> &nn_arch, 
// 		const std::string act_func, 
// 		const unsigned conv_n_filter,
// 		const unsigned conv_Hf,
// 		std::vector<MyMatrix> &W,
// 		std::vector<MyVector> &B,
// 		MyMatrix &conv_W,
// 		MyVector &conv_B){
  
//   // Fix some parameters
//   const unsigned n_layers = nn_arch.size();
//   double sigma = 1.0;
//   if(act_func=="sigmoid"){
//     sigma = 4.0;
//   }
//   unsigned param_counter = 0;

//   // Initialize the convolutional layer
//   initConvLayer(conv_n_filter, conv_Hf, conv_W, conv_B);
  
//   // Initialize the weights
//   for(unsigned i = 0; i < n_layers-1; i++){
//     initLayer(nn_arch[i], nn_arch[i+1], sigma, W[i]);
//     if(act_func=="sigmoid"){
//       B[i] = -0.5 * W[i].colwise().sum();
//     }else if(act_func=="tanh"){
//       B[i] = MyVector::Zero(W[i].cols());
//     }else if(act_func=="relu"){ // TODO: find the right way to initialize relu
//       B[i] = MyVector::Zero(W[i].cols());
//     }else{
//       std::cout << "Not implemented!" << std::endl;
//       assert(false);
//     }
//     param_counter += nn_arch[i] * nn_arch[i+1] + nn_arch[i+1];
//   }
//   return param_counter;
// }


double signFunc(double x){
  if(x<0.0)
    return -1.0;
  else if(x>0.0)
    return 1.0;
  else
    return 0.0;
}

double squareFunc(double x){
  return x*x;
}

double sqrtFunc(double x){
  return sqrt(x);
}

double sumCstFunc(double x, double cst){
  return x+cst;
}

void softmax(const MyMatrix &a, 
	     MyMatrix& out){
  
  MyVector max_coeff = a.rowwise().maxCoeff();
  MyMatrix centered_a = a.colwise() - max_coeff;
  
  // Compute the exponential of the entries
  for(unsigned j = 0; j < a.cols(); j++){ // Optimization for column-major
    for(unsigned i = 0; i < a.rows(); i++){  
      centered_a(i,j) = exp(centered_a(i,j));
    }
  }
  MyVector sum_coeff = centered_a.rowwise().sum();
  out = centered_a.array().colwise() / sum_coeff.array();
}

/// Activation functions
void logistic(const bool deriv_flag,
	      const MyMatrix &z, 
	      MyMatrix &a, 
	      MyMatrix &ap){
  if(deriv_flag){
    for(int j = 0; j < z.cols(); j++){ // Optimization for column-major
      for(int i = 0; i < z.rows(); i++){
	a(i,j) = 1.0/(1.0+exp(-z(i,j)));
	ap(i,j) = a(i,j) * (1.0-a(i,j));
      }
    }
  }else{
    for(int j = 0; j < z.cols(); j++){ // Optimization for column-major
      for(int i = 0; i < z.rows(); i++){
	a(i,j) = 1.0/(1.0+exp(-z(i,j)));
      }
    }
  }
}

void my_tanh(const bool deriv_flag,
	     const MyMatrix &z, 
	     MyMatrix &a, 
	     MyMatrix &ap){
  if(deriv_flag){
    for(int j = 0; j < z.cols(); j++){ // Optimization for column-major
      for(int i = 0; i < z.rows(); i++){
	a(i,j) = tanh(z(i,j));
	ap(i,j) = (1.0-a(i,j)*a(i,j));
      }
    }
  }else{
    for(int j = 0; j < z.cols(); j++){ // Optimization for column-major
      for(int i = 0; i < z.rows(); i++){
	a(i,j) = tanh(z(i,j));
      }
    }
  }
}

void relu(const bool deriv_flag,
	  const MyMatrix &z, 
	  MyMatrix &a, 
	  MyMatrix &ap){
  if(deriv_flag){
    for(int j = 0; j < z.cols(); j++){ // Optimization for column-major
      for(int i = 0; i < z.rows(); i++){
	a(i,j) = max(0.0,z(i,j));
	if(z(i,j)>0)
	  ap(i,j) = 1.0;
	else
	  ap(i,j) = 0.0;
      }
    }
  }else{
    for(int j = 0; j < z.cols(); j++){ // Optimization for column-major
      for(int i = 0; i < z.rows(); i++){
	a(i,j) = max(0.0,z(i,j));
      }
    }
  }
}
//////////////////

void dropout(const double prob,
	     MyMatrix &A,
	     MyMatrix &B){

  // Create the mask
  std::bernoulli_distribution dist(prob);
  MyMatrix mask(A.rows(),A.cols());
  for(unsigned j = 0; j < A.cols(); j++){
    for(unsigned i = 0; i < A.rows(); i++){
      mask(i,j) = dist(gen);
    }
  }
  
  // Multiply element-wise
  A = A.cwiseProduct(mask);
  B = B.cwiseProduct(mask);
}

void dropout(const double prob,
	     MyMatrix &A){

  // Create the mask
  std::bernoulli_distribution dist(prob);
  MyMatrix mask(A.rows(),A.cols());
  for(unsigned j = 0; j < A.cols(); j++){
    for(unsigned i = 0; i < A.rows(); i++){
      mask(i,j) = dist(gen);
    }
  }

  // Multiply element-wise
  A = A.cwiseProduct(mask);
}

void buildConvMatrix(const unsigned n_img,
		     const unsigned conv_N1,
		     const unsigned conv_N2,
		     const unsigned F,
		     const unsigned S,
		     const unsigned P,
		     const MyMatrix& conv_layer,
		     MyMatrix& conv_matrix){

  const unsigned conv_depth = conv_layer.rows();
  conv_matrix.setZero(conv_depth*F*F,n_img*conv_N2*conv_N2);
  for(unsigned i = 0; i < n_img; i++){
    for(unsigned j = 0; j < conv_depth; j++){
      for(unsigned no = 0; no < conv_N2; no++){
	for(unsigned ni = 0; ni < conv_N2; ni++){
	  const unsigned ko_start = std::max(0,(int)P-(int)no);
	  const unsigned ko_end = F-std::max(0,(int)P-(int)conv_N2+(int)no+1);
	  const unsigned ki_start = std::max(0,(int)P-(int)ni);
	  const unsigned ki_end = F-std::max(0,(int)P-(int)conv_N2+(int)ni+1);
	  for(unsigned ko = ko_start; ko < ko_end; ko++){
	    for(unsigned ki = ki_start; ki < ki_end; ki++){
	      const unsigned idx = ki + ko * conv_N1 + (ni-P) * S + (no-P) * S * conv_N1 + i*conv_N1*conv_N1;
	      conv_matrix(j*F*F+ko*F+ki,no*conv_N2+ni+i*conv_N2*conv_N2) = conv_layer(j,idx);
	    }
	  }
	}
      }
    }
  }
}

void poolMax(const unsigned n_img,
	     const unsigned conv_N1,
	     const unsigned conv_N2,
	     const unsigned F,
	     const unsigned S,
	     const MyMatrix& conv_layer,
	     MyMatrix& pool_layer,
	     std::vector<unsigned> &pool_idx_x,
	     std::vector<unsigned> &pool_idx_y){

  const unsigned conv_depth = conv_layer.rows();
  pool_layer.resize(conv_depth, conv_N2*conv_N2*n_img);
  
  for(unsigned i = 0; i < n_img; i++){
    for(unsigned j = 0; j < conv_depth; j++){
      for(unsigned no = 0; no < conv_N2; no++){
	for(unsigned ni = 0; ni < conv_N2; ni++){
	  double max_val = -1.0e20;
	  unsigned max_idx = -1;
	  for(unsigned ko = 0; ko < F; ko++){
	    for(unsigned ki = 0; ki < F; ki++){
	      const unsigned idx = ki + ko * conv_N1 + ni * S + no * S * conv_N1 + i*conv_N1*conv_N1;
	      if(conv_layer(j,idx) > max_val){
		max_val = conv_layer(j,idx);
		max_idx = idx;
	      }
	    }
	  }
	  const unsigned idx2 = i*conv_N2*conv_N2+no*conv_N2+ni;
	  pool_layer(j,idx2) = max_val;
	  pool_idx_x.push_back(j);
	  pool_idx_y.push_back(max_idx);
	}
      }
    }
  }
}

void fprop(const bool dropout_flag,
	   const ActivationFunction &act_func,
	   const std::vector<MySpMatrix> &W,
	   const MyMatrix &W_out,
	   const std::vector<MyVector> &B,
	   const MyMatrix &X_batch,
	   std::vector<MyMatrix> &Z,
	   std::vector<MyMatrix> &A,
	   std::vector<MyMatrix> &Ap){

  const MyMatrix* Atilde = &X_batch;
  if(dropout_flag){
    Atilde = new MyMatrix(X_batch);
    dropout(0.8, *const_cast<MyMatrix *>(Atilde));
  }

  for(unsigned i = 0; i < A.size(); i++){
    Z[i] = (*Atilde * W[i]).rowwise() + B[i].transpose();
    A[i].resize(Z[i].rows(), Z[i].cols());
    Ap[i].resize(Z[i].rows(), Z[i].cols());
    act_func(Z[i],A[i],Ap[i]);
    if(dropout_flag){
      dropout(0.5, A[i], Ap[i]);
      if(i==0)
	delete Atilde;
    }
    Atilde = &A[i];
  }

  // Output layer
  Z[A.size()] = (*Atilde * W_out).rowwise() + B[A.size()].transpose();

}

void fprop(const bool dropout_flag,
	   const ActivationFunction &act_func,
	   const std::vector<MyMatrix> &W, 
	   const std::vector<MyVector> &B,
	   const MyMatrix &X_batch,
	   std::vector<MyMatrix> &Z,
	   std::vector<MyMatrix> &A,
	   std::vector<MyMatrix> &Ap){
  
  const MyMatrix* Atilde = &X_batch;
  if(dropout_flag){
    Atilde = new MyMatrix(X_batch);
    dropout(0.8, *const_cast<MyMatrix *>(Atilde));
  }
  
  for(unsigned i = 0; i < A.size(); i++){
    Z[i] = (*Atilde * W[i]).rowwise() + B[i].transpose();
    A[i].resize(Z[i].rows(), Z[i].cols());
    Ap[i].resize(Z[i].rows(), Z[i].cols());
    act_func(Z[i],A[i],Ap[i]);
    if(dropout_flag){
      dropout(0.5, A[i], Ap[i]);
      if(i==0)
	delete Atilde;
    }
    Atilde = &A[i];
  }

  // Output layer
  Z[A.size()] = (A[A.size()-1] * W[A.size()]).rowwise() + B[A.size()].transpose();
}


void convFprop(const unsigned batch_size,
	       const std::vector<ConvLayerParams> &conv_params,
	       const std::vector<PoolLayerParams> &pool_params,
	       const ActivationFunction &act_func,
	       const std::vector<MyMatrix> &conv_W, 
	       const std::vector<MyVector> &conv_B,
	       const MyMatrix &X_batch,
	       std::vector<MyMatrix> &conv_A,
	       std::vector<MyMatrix> &conv_Ap,
	       MyMatrix& z0,
	       std::vector<std::vector<unsigned>> &poolIdxX,
	       std::vector<std::vector<unsigned>> &poolIdxY){

  const MyMatrix* Atilde = &X_batch;
  MyMatrix pool_z0;
  for(unsigned i = 0; i < conv_W.size(); i++){

    MyMatrix conv_z0 = conv_W[i] * *Atilde;
    conv_z0.colwise() += conv_B[i];

    MyMatrix conv_a(conv_z0.rows(), conv_z0.cols());;
    conv_Ap[i].resize(conv_z0.rows(), conv_z0.cols());

    act_func(conv_z0,conv_a,conv_Ap[i]);
  
    // Pool max for conv layer
    poolMax(batch_size, conv_params[i].N, pool_params[i].N, pool_params[i].Hf, pool_params[i].stride, conv_a, pool_z0, poolIdxX[i], poolIdxY[i]);
    
    if(i < conv_W.size()-1){
      buildConvMatrix(batch_size, pool_params[i].N, conv_params[i+1].N, conv_params[i+1].Hf, conv_params[i+1].stride, conv_params[i+1].padding, pool_z0, conv_A[i]);
      Atilde = &conv_A[i];
    }
  }

  // Convert the pool layer into a FNN layer
  conv2Layer(batch_size, pool_params[conv_W.size()-1].N, pool_z0, z0);
}

void bprop(const std::vector<MyMatrix> &W,
	   const std::vector<MyMatrix> &Ap,
	   std::vector<MyMatrix> &gradB){
  
  const unsigned n_layers = W.size() + 1;
  for(unsigned i = 0; i < W.size()-1; i++){
    const unsigned rev_i = n_layers - i - 3;
    gradB[rev_i] = (gradB[rev_i+1] * W[rev_i+1].transpose()).cwiseProduct(Ap[rev_i]);
  }
}

void bprop(const std::vector<MySpMatrix> &Wt,
	   const MyMatrix &W_out,
	   const std::vector<MyMatrix> &Ap,
	   std::vector<MyMatrix> &gradB){

  const unsigned n_layers = Wt.size() + 2;
  gradB[n_layers-3] = (gradB[n_layers-2] * W_out.transpose()).cwiseProduct(Ap[n_layers-3]);

  for(unsigned i = 0; i < Wt.size()-1; i++){
    const unsigned rev_i = n_layers - i - 4;
    gradB[rev_i] = (gradB[rev_i+1] * Wt[rev_i+1]).cwiseProduct(Ap[rev_i]);
  }
}

void convBprop(const unsigned batch_size,
	       const std::vector<ConvLayerParams> &conv_params,
	       const std::vector<PoolLayerParams> &pool_params,
	       const std::vector<MyMatrix> &convW_T,
	       const std::vector<MyMatrix> &convAp,
	       const MyMatrix &pool_gradB,
	       std::vector<MyMatrix> &gradB,
	       std::vector<std::vector<unsigned>> &poolIdxX,
	       std::vector<std::vector<unsigned>> &poolIdxY){
  
  const MyMatrix* Atilde = &pool_gradB;
  
  const unsigned n_layers = convW_T.size();
  for(unsigned l = 0; l < n_layers; l++){
    
    const unsigned rev_l = n_layers - l - 1;
    
    MyMatrix conv_gradB_act;
    pool2conv(batch_size, conv_params[rev_l].n_filter, conv_params[rev_l].N, pool_params[rev_l].N, poolIdxX[rev_l], poolIdxY[rev_l], *const_cast<MyMatrix *>(Atilde), conv_gradB_act);
    
    gradB[rev_l] = conv_gradB_act.cwiseProduct(convAp[rev_l]);
    
    if(rev_l>0){
      MyMatrix conv_mat_pool;
      buildConvMatrix(batch_size, conv_params[rev_l].N, pool_params[rev_l-1].N, conv_params[rev_l].Hf, conv_params[rev_l].stride, pool_params[rev_l-1].N-conv_params[rev_l].N, gradB[rev_l], conv_mat_pool);
      
      *const_cast<MyMatrix *>(Atilde) = convW_T[rev_l] * conv_mat_pool;
    }
  }
}

void qdBpmBprop(const std::vector<MyMatrix> &W,
		const std::vector<MyMatrix> &Ap,
		std::vector<MyMatrix> &bp_gradB){
  
  const unsigned n_layers = W.size() + 1;
  for(unsigned i = 0; i < W.size()-1; i++){
    const unsigned rev_i = n_layers - i - 3;
    MyMatrix W_sq = W[rev_i+1].unaryExpr(std::ptr_fun(squareFunc));
    bp_gradB[rev_i] = (bp_gradB[rev_i+1] * W_sq.transpose()).cwiseProduct(Ap[rev_i].array().square().matrix());
  }
}

void sparseOuterProduct(const unsigned batch_size,
			const MySpMatrix &W,
			const MyMatrix &gradB,
			const MyMatrix &A,
			const double mat_reg,
			MySpMatrix &dw){

  unsigned l = 0;
  std::vector<SpEntry> coeffdW(W.nonZeros());
  for (unsigned k=0; k < W.outerSize(); k++){
    for (MySpMatrix::InnerIterator it(W,k); it; ++it){
      const double val = A.col(it.row()).dot(gradB.col(it.col()));
      coeffdW[l++] = SpEntry(it.row(), it.col(), val/batch_size + mat_reg);
    }
  }
  dw.setFromTriplets(coeffdW.begin(), coeffdW.end());
}


template <class T>
void updateParam(const double eta,
		 const std::string regularizer,
		 const double lambda,
		 const T &dparams,
		 T &params){
  
  if(regularizer==""){
    params -= eta * dparams;
  }else if(regularizer=="L1"){
    T sign_params = params.unaryExpr(std::ptr_fun(signFunc));
    params = params - eta * lambda * sign_params - eta * dparams;
  }else if(regularizer=="L2"){
    params = (1.0-eta * lambda)*params - eta * dparams;
  }
  else{
    std::cout << "Not implemented" << std::endl;
    assert(false);
  }
}

void updateLayer(const double eta,
		 const unsigned batch_size,
		 const MyMatrix &gradB,
		 const MyMatrix &A,
		 const std::string regularizer,
		 const double lambda,
		 MySpMatrix &W,
		 MySpMatrix &Wt,
		 MyVector &B){

  MySpMatrix dw(W.rows(),W.cols());
  sparseOuterProduct(batch_size,W,gradB,A,0.0,dw);

  updateParam(eta, regularizer, lambda, dw, W);
  MySpMatrix dwt = dw.transpose();
  updateParam(eta, regularizer, lambda, dwt, Wt);

  MyVector gradB_avg = gradB.colwise().sum() / batch_size;
  updateParam(eta, regularizer, lambda, gradB_avg, B);

}



void update(const double eta,
	    const std::vector<MyMatrix> &gradB,
	    const std::vector<MyMatrix> &A,
	    const MyMatrix &X_batch,
	    const std::string regularizer,
	    const double lambda,
	    MyMatrix &W_out,
	    std::vector<MySpMatrix> &W,
	    std::vector<MySpMatrix> &Wt,
	    std::vector<MyVector> &B){

  const unsigned batch_size = X_batch.rows();
  const unsigned n_layers = W.size()+2;

  // Update the first layer
  updateLayer(eta, batch_size, gradB[0], X_batch, regularizer, lambda, W[0], Wt[0], B[0]);

  // Update the last layer
  const MyMatrix dw_out = (A[n_layers-3].transpose() * gradB[n_layers-2])/batch_size;
  updateParam(eta, regularizer, lambda, dw_out, W_out);

  const MyVector gradB_avg1 = gradB[n_layers-2].colwise().sum() / batch_size;
  updateParam(eta, regularizer, lambda, gradB_avg1, B[n_layers-2]);

  // Update the hidden layers
  for(unsigned i = 1; i < W.size(); i++){
    updateLayer(eta, batch_size, gradB[i], A[i-1], regularizer, lambda, W[i], Wt[i], B[i]);
  }
}



void update(const double eta,
	    const std::vector<MyMatrix> &gradB, 
	    const std::vector<MyMatrix> &A, 
	    const MyMatrix &X_batch,
	    const std::string regularizer, 
	    const double lambda,
	    std::vector<MyMatrix> &W,
	    std::vector<MyVector> &B){
  
  const unsigned batch_size = X_batch.rows(); 
  
  // Update the hidden layers
  for(unsigned i = 0; i < W.size(); i++){

    const MyMatrix* Atilde = &X_batch;
    if(i > 0){
      Atilde = &A[i-1];
    }

    MyMatrix dw = (Atilde->transpose() * gradB[i])/batch_size;
    updateParam(eta, regularizer, lambda, dw, W[i]);
    const MyVector gradB_avg = gradB[i].colwise().sum()/batch_size;
    updateParam(eta, regularizer, lambda, gradB_avg, B[i]);
  }
}

void convUpdate(const double eta,
		const std::vector<MyMatrix> &conv_gradB, 
		const std::vector<MyMatrix> &conv_A, 
		const MyMatrix &X_batch,
		const std::string regularizer,
		const double lambda,
		std::vector<MyMatrix> &conv_W,
		std::vector<MyVector> &conv_B){

  for(unsigned l = 0; l < conv_W.size(); l++){
    const MyMatrix* Atilde = &X_batch;
    if(l > 0){
      Atilde = &conv_A[l-1];
    }

    MyMatrix conv_update = conv_gradB[l] * Atilde->transpose();
    conv_W[l] -= eta * conv_update;
    conv_B[l] -= eta * conv_update.rowwise().sum();
  }
}

void convUpdate(const unsigned batch_size,
		const double eta,
		const std::vector<ConvLayerParams> &conv_params,
		const std::vector<MyMatrix> &conv_gradB, 
		const std::vector<MyMatrix> &conv_A, 
		const MyMatrix &X_batch,
		const std::string regularizer,
		const double lambda,
		std::vector<MyMatrix> &conv_W,
		std::vector<MyMatrix> &conv_W_T,
		std::vector<MyVector> &conv_B){

  for(unsigned l = 0; l < conv_W.size(); l++){
    const MyMatrix* Atilde = &X_batch;
    if(l > 0){
      Atilde = &conv_A[l-1];
    }

    MyMatrix conv_update = conv_gradB[l] * Atilde->transpose();
    conv_W[l] -= eta * conv_update / batch_size;

    const unsigned Hf = conv_params[l].Hf;
    unsigned n_channel = N_CHANNEL;
    if(l>0){
      n_channel = conv_params[l-1].n_filter;
    }

    transposeConvW(conv_W[l],n_channel, Hf,conv_W_T[l]);
    conv_B[l] -= eta * conv_gradB[l].rowwise().sum() / batch_size;
  }
}

void convUpdateTest(const unsigned batch_size,
		    const double eta,
		    const std::vector<MyMatrix> &conv_gradB, 
		    const std::vector<MyMatrix> &conv_A, 
		    const MyMatrix &X_batch,
		    const std::string regularizer,
		    const double lambda,
		    std::vector<MyMatrix> &conv_W,
		    std::vector<MyVector> &conv_B,
		    std::vector<MyMatrix> &conv_update,
		    std::vector<MyVector> &conv_updateB){
  
  for(unsigned l = 0; l < conv_W.size(); l++){
    const MyMatrix* Atilde = &X_batch;
    if(l > 0){
      Atilde = &conv_A[l-1];
    }

    conv_update[l] = (conv_gradB[l] * Atilde->transpose())/batch_size;
    conv_updateB[l] = conv_gradB[l].rowwise().sum()/batch_size;
    // conv_W[l] -= eta * conv_update;
    // conv_B[l] -= eta * conv_update.rowwise().sum();
  }
}

void testUpdate(const double eta,
		const std::vector<MyMatrix> &gradB, 
		const std::vector<MyMatrix> &A, 
		const MyMatrix &X_batch,
		const std::string regularizer, 
		const double lambda,
		std::vector<MyMatrix> &W,
		std::vector<MyVector> &B,
		std::vector<MyMatrix> &DW,
		std::vector<MyVector> &DB){
  
  const unsigned batch_size = X_batch.rows(); 
  const unsigned n_layers = W.size()+1;
  
  DW.resize(W.size());
  DB.resize(B.size());

  // Update the hidden layers
  for(unsigned i = 0; i < W.size(); i++){

    const MyMatrix* Atilde = &X_batch;
    if(i > 0){
      Atilde = &A[i-1];
    }

    MyMatrix dw = (Atilde->transpose() * gradB[i])/batch_size;
    //updateParam(eta, regularizer, lambda, dw, W[i]);
    const MyVector gradB_avg = gradB[i].colwise().sum()/batch_size;
    //updateParam(eta, regularizer, lambda, gradB_avg, B[i]);

    DW[i] = dw;
    DB[i] = gradB_avg;
  }
}

void adagradUpdate(const double eta,
		   const std::vector<MyMatrix> &gradB,
		   const std::vector<MyMatrix> &A, 
		   const MyMatrix &X_batch,
		   const std::string regularizer, 
		   const double lambda,
		   const double mat_reg,
		   const double autocorr,
		   std::vector<MyMatrix> &W,
		   std::vector<MyVector> &B,
		   std::vector<MyMatrix> &mu_dW,
		   std::vector<MyVector> &mu_dB){
  
  const unsigned batch_size = X_batch.rows(); 

  bool ada_first = true;
  if(mu_dW[0].size()!=0)
    ada_first = false;
  
  // Update the hidden layers
  for(unsigned i = 0; i < W.size(); i++){

    const MyMatrix* Atilde = &X_batch;
    if(i > 0){
      Atilde = &A[i-1];
    }

    const MyMatrix dw = (Atilde->transpose() * gradB[i])/batch_size;
    if(ada_first){
      mu_dW[i] = dw.array().square();
    }else{
      mu_dW[i] = autocorr * mu_dW[i].array() + (1.0-autocorr) * dw.array().square();
    }
    
    MyMatrix temp = mu_dW[i].cwiseSqrt().array() + mat_reg;
    MyMatrix ada_dw = dw.cwiseQuotient(temp);
    
    updateParam(eta, regularizer, lambda, ada_dw, W[i]);

    const MyVector gradB_avg = gradB[i].colwise().sum() / batch_size;
    if(ada_first){
      mu_dB[i] = gradB_avg.array().square();
    }else{
      mu_dB[i] = autocorr * mu_dB[i].array() + (1.0-autocorr) * gradB_avg.array().square();
    }
    
    MyVector temp2 = mu_dB[i].cwiseSqrt().array() + mat_reg;
    MyVector ada_dB = gradB_avg.cwiseQuotient(temp2);
    updateParam(eta, regularizer, lambda, ada_dB, B[i]);
  }
}

void computeMcError(const MyMatrix &out,
		    MyMatrix &mc_error){

  MyMatrix mc_one_hot = MyMatrix::Zero(out.rows(), out.cols());

  for(unsigned i = 0; i < out.rows(); i++){
    
    std::vector<double> probs(out.cols());
    for(unsigned j = 0; j < out.cols(); j++){
      probs[j] = out(i,j);
    }
    std::discrete_distribution<int> dist(probs.begin(), probs.end());

    const unsigned label = dist(gen);
    mc_one_hot(i,label) = 1.0;
  }
  mc_error = out - mc_one_hot;
}

void computeLazyError(const MyMatrix &out,
		      const MyMatrix &one_hot_batch,
		      MyMatrix &lazy_error){
  
  const unsigned n_labels = out.cols();
  
  lazy_error.setZero(out.rows(),out.cols());
  
  for(unsigned i = 0; i < out.rows(); i++){
    for(unsigned j = 0; j < out.cols(); j++){
      if(one_hot_batch(i,j)==1){
	lazy_error(i,j) = (out(i,j)-1.0) * out(i,j);
      }
      else{
	lazy_error(i,j) = out(i,j) * out(i,j);
      }
    }
  }
}

void updateConvMetric(const bool init_flag,
		      const double gamma,
		      std::vector<MyMatrix> &conv_Mii,
  		      std::vector<MyMatrix> &conv_M0i,
  		      std::vector<MyVector> &conv_M00,
		      std::vector<MyMatrix> &conv_pMii,
		      std::vector<MyMatrix> &conv_pM0i,
		      std::vector<MyVector> &conv_pM00){
  if(!init_flag){
    for(unsigned k = 0; k < conv_Mii.size(); k++){
      conv_Mii[k] = (1.0 - gamma) * conv_pMii[k]  + gamma * conv_Mii[k];
      conv_M0i[k] = (1.0 - gamma) * conv_pM0i[k]  + gamma * conv_M0i[k];
      conv_M00[k] = (1.0 - gamma) * conv_pM00[k]  + gamma * conv_M00[k];
    }
  }
  conv_pMii = conv_Mii;
  conv_pM0i = conv_M0i;
  conv_pM00 = conv_M00;
}

void updateMetric(const bool init_flag,
		  const double gamma,
		  MyMatrix &Mii_out,
		  MyMatrix &M0i_out,
		  MyVector &M00_out,
		  std::vector<MySpMatrix> &Mii,
		  std::vector<MySpMatrix> &M0i,
		  std::vector<MyVector> &M00,
		  MyMatrix &pMii_out,
		  MyMatrix &pM0i_out,
		  MyVector &pM00_out,
		  std::vector<MySpMatrix> &pMii,
		  std::vector<MySpMatrix> &pM0i,
		  std::vector<MyVector> &pM00){

  // Update the metric
  if(!init_flag){
    Mii_out = (1.0 - gamma) * pMii_out  + gamma * Mii_out;
    M0i_out = (1.0 - gamma) * pM0i_out  + gamma * M0i_out;
    M00_out = (1.0 - gamma) * pM00_out  + gamma * M00_out;

    for(unsigned k = 0; k < Mii.size(); k++){
      Mii[k] = (1.0 - gamma) * pMii[k]  + gamma * Mii[k];
      M0i[k] = (1.0 - gamma) * pM0i[k]  + gamma * M0i[k];
      M00[k] = (1.0 - gamma) * pM00[k]  + gamma * M00[k];
    }
  }
  pMii_out = Mii_out;
  pM0i_out = M0i_out;
  pM00_out = M00_out;

  pMii = Mii;
  pM0i = M0i;
  pM00 = M00;
}

void updateMetric(const bool init_flag, 
		  const double gamma,
		  std::vector<MyMatrix> &Mii, 
		  std::vector<MyMatrix> &M0i, 
		  std::vector<MyVector> &M00, 
		  std::vector<MyMatrix> &pMii,
		  std::vector<MyMatrix> &pM0i,
		  std::vector<MyVector> &pM00){
  
  // Update the metric
  if(!init_flag){	
    for(unsigned k = 0; k < Mii.size(); k++){
      Mii[k] = (1.0 - gamma) * pMii[k]  + gamma * Mii[k];
      M0i[k] = (1.0 - gamma) * pM0i[k]  + gamma * M0i[k];
      M00[k] = (1.0 - gamma) * pM00[k]  + gamma * M00[k];
    }
  }      
  pMii = Mii;
  pM0i = M0i;
  pM00 = M00;
}

void updateMetric(const bool init_flag, 
		  const double gamma,
		  std::vector<MyMatrix> &Mii, 
		  std::vector<MyVector> &M00, 
		  std::vector<MyMatrix> &pMii,
		  std::vector<MyVector> &pM00){
  
  // Update the metric
  if(!init_flag){	
    for(unsigned k = 0; k < Mii.size(); k++){
      Mii[k] = (1.0 - gamma) * pMii[k]  + gamma * Mii[k];
      M00[k] = (1.0 - gamma) * pM00[k]  + gamma * M00[k];
    }
  }      
  pMii = Mii;
  pM00 = M00;
}

void buildConvQDMetric(const unsigned batch_size,
		       const std::vector<MyMatrix> &conv_gradB_sq,
		       const std::vector<MyMatrix> &conv_A,
		       const MyMatrix &X_batch,
		       const std::vector<MyMatrix> &conv_W, 
		       const double mat_reg,
		       std::vector<MyMatrix> &conv_Mii,
		       std::vector<MyMatrix> &conv_M0i,
		       std::vector<MyVector> &conv_M00){

  const MyMatrix* Atilde = &X_batch;  
  for(unsigned i = 0; i < conv_Mii.size(); i++){
    const MyMatrix A_sq = Atilde->array().square();
    conv_Mii[i] =  conv_gradB_sq[i] * A_sq.transpose() / batch_size;
    conv_Mii[i].array() += mat_reg;
    conv_M0i[i] = conv_gradB_sq[i] * Atilde->transpose() / batch_size;
    conv_M00[i] = conv_gradB_sq[i].rowwise().sum() / batch_size;
    conv_M00[i].array() += mat_reg;

    Atilde = &conv_A[i];

  }
}

void buildQDMetric(const std::vector<MyMatrix> &gradB_sq,
		   const std::vector<MyMatrix> &A,
		   const MyMatrix &X_batch,
		   const MyMatrix &W_out,
		   const std::vector<MySpMatrix> &W,
		   const double mat_reg,
		   MyMatrix &Mii_out,
		   MyMatrix &M0i_out,
		   MyVector &M00_out,
		   std::vector<MySpMatrix> &Mii,
		   std::vector<MySpMatrix> &M0i,
		   std::vector<MyVector> &M00){

  const unsigned batch_size = X_batch.rows();
  const MyMatrix* Atilde = &X_batch;

  for(unsigned i = 0; i < W.size(); i++){

    Mii[i].resize(W[i].rows(),W[i].cols());
    const MyMatrix A_sq = Atilde->array().square();
    sparseOuterProduct(batch_size,W[i],gradB_sq[i],A_sq,mat_reg,Mii[i]);

    M0i[i].resize(W[i].rows(),W[i].cols());
    sparseOuterProduct(batch_size,W[i],gradB_sq[i],*Atilde,0.0,M0i[i]);

    M00[i] = gradB_sq[i].colwise().sum() / batch_size;  // Bug in normalization
    M00[i].array() += mat_reg;
    Atilde = &A[i];

  }

  // Compute the metric of the last layer
  const unsigned n_layers = W.size()+2;
  Mii_out.resize(W_out.rows(),W_out.cols());
  MyMatrix A_sq = Atilde->array().square();
  Mii_out = (A_sq.transpose() * gradB_sq[n_layers-2]) / batch_size;
  Mii_out.array() += mat_reg;

  M0i_out.resize(W_out.rows(),W_out.cols());
  M0i_out = (Atilde->transpose() * gradB_sq[n_layers-2]) / batch_size;

  M00_out = gradB_sq[n_layers-2].colwise().sum() / batch_size;
  M00_out.array() += mat_reg;

}

void buildQDMetric(const std::vector<MyMatrix> &gradB_sq, 
		   const std::vector<MyMatrix> &A, 
		   const MyMatrix &X_batch,
		   const std::vector<MyMatrix> &W, 
		   const double mat_reg,
		   std::vector<MyMatrix> &Mii,
		   std::vector<MyMatrix> &M0i,
		   std::vector<MyVector> &M00){
  
  const unsigned batch_size = X_batch.rows(); 
  const MyMatrix* Atilde = &X_batch;
  
  for(unsigned i = 0; i < W.size(); i++){
    Mii[i].resize(W[i].rows(),W[i].cols());
    const MyMatrix A_sq = Atilde->array().square();
    Mii[i] = (A_sq.transpose() * gradB_sq[i]) / batch_size;
    Mii[i].array() += mat_reg;
    
    M0i[i] = (Atilde->transpose() * gradB_sq[i]) / batch_size;
    
    M00[i] = gradB_sq[i].colwise().sum() / batch_size;
    M00[i].array() += mat_reg;
    Atilde = &A[i];

  }
}

void buildDiagMetric(const std::vector<MyMatrix> &gradB_sq, 
		     const std::vector<MyMatrix> &A, 
		     const MyMatrix &X_batch,
		     const std::vector<MyMatrix> &W, 
		     const double mat_reg,
		     std::vector<MyMatrix> &Mii,
		     std::vector<MyVector> &M00){
  
  const unsigned batch_size = X_batch.rows(); 
  const MyMatrix* Atilde = &X_batch;
  
  for(unsigned i = 0; i < W.size(); i++){
    Mii[i].resize(W[i].rows(),W[i].cols());
    const MyMatrix A_sq = Atilde->array().square();
    Mii[i] = (A_sq.transpose() * gradB_sq[i]) / batch_size;
    Mii[i].array() += mat_reg;
        
    M00[i] = gradB_sq[i].colwise().sum() / batch_size;
    M00[i].array() += mat_reg;
    Atilde = &A[i];
  }
}

void qdGradient(const MyMatrix &Mii,
		const MyMatrix &M0i,
		const MyVector &M00,
		const MyMatrix &G,
		const MyVector &G0,
		MyMatrix &qd_gradW,
		MyVector &qd_dBias){

  double temp = 0.0;
  for(unsigned k = 0; k < G.cols(); k++){ 
    for(unsigned i = 0; i < G.rows(); i++){
      qd_gradW(i,k) = (G(i,k) * M00(k) - G0(k) * M0i(i,k)) / ((Mii(i,k) * M00(k)) - (M0i(i,k) * M0i(i,k)));
      temp += (M0i(i,k) / M00(k)) * qd_gradW(i,k);
    }
    qd_dBias(k) = (G0(k) / M00(k)) - temp;
    temp = 0.0;
  }
}

void qdConvGradient(const MyMatrix &Mii,
		    const MyMatrix &M0i,
		    const MyVector &M00,
		    const MyMatrix &G,
		    const MyVector &G0,
		    MyMatrix &qd_gradW,
		    MyVector &qd_dBias){
  
  double temp = 0.0;
  for(unsigned k = 0; k < G.rows(); k++){ 
    for(unsigned i = 0; i < G.cols(); i++){
      qd_gradW(k,i) = (G(k,i) * M00(k) - G0(k) * M0i(k,i)) / ((Mii(k,i) * M00(k)) - (M0i(k,i) * M0i(k,i)));
      temp += (M0i(k,i) / M00(k)) * qd_gradW(k,i);
    }
    qd_dBias(k) = (G0(k) / M00(k)) - temp;
    temp = 0.0;
  }
}

void diagGradient(const MyMatrix &Mii,
		  const MyVector &M00,
		  const MyMatrix &G,
		  const MyVector &G0,
		  MyMatrix &qd_gradW,
		  MyVector &qd_dBias){

  double temp = 0.0;
  for(unsigned k = 0; k < G.cols(); k++){ 
    for(unsigned i = 0; i < G.rows(); i++){
      qd_gradW(i,k) = G(i,k)/Mii(i,k);
    }
    qd_dBias(k) = G0(k)/M00(k);
    temp = 0.0;
  }
}

void qdSpGradient(const MySpMatrix &W,
		  const MySpMatrix &Mii,
		  const MySpMatrix &M0i,
		  const MyVector &M00,
		  const MySpMatrix &G,
		  const MyVector &G0,
		  MySpMatrix &qd_gradW,
		  MyVector &qd_dBias){

  unsigned l = 0;
  double temp = 0.0;
  std::vector<SpEntry> coeffdW(W.nonZeros());
  for (unsigned k=0; k < W.outerSize(); k++){
    for (MySpMatrix::InnerIterator it(G,k), it2(Mii,k), it3(M0i,k); it; ++it,++it2,++it3){ // Synchronize all the iterators for fast access
      const double Gik = it.value();
      const double Miik = it2.value();
      const double M0ik = it3.value();
      const double val = (Gik * M00(k) - G0(k) * M0ik) / ((Miik * M00(k)) - (M0ik * M0ik));
      coeffdW[l++] = SpEntry(it.row(), it.col(), val);
      
      temp += (M0ik / M00(k)) * val;
    }
    qd_dBias(k) = (G0(k) / M00(k)) - temp;
    temp = 0.0;
  }
  qd_gradW.setFromTriplets(coeffdW.begin(), coeffdW.end());
}


void opUpdateLayer(const double eta,
		   const unsigned batch_size,
		   const MyMatrix &gradB,
		   const MyMatrix &A,
		   const std::string regularizer,
		   const double lambda,
		   const MySpMatrix &Mii,
		   const MySpMatrix &M0i,
		   const MyVector &M00,
		   MySpMatrix &W,
		   MySpMatrix &Wt,
		   MyVector &B){

  MySpMatrix dw(W.rows(),W.cols());
  sparseOuterProduct(batch_size,W,gradB,A,0.0,dw);
  MyVector gradB_avg = gradB.colwise().sum() / batch_size;

  MySpMatrix qd_dw(dw.rows(), dw.cols());
  MyVector qd_dBias(B.size());
  qdSpGradient(W, Mii, M0i, M00, dw, gradB_avg, qd_dw, qd_dBias);

  updateParam(eta, regularizer, lambda, qd_dw, W);
  MySpMatrix qd_dwt = qd_dw.transpose();
  updateParam(eta, regularizer, lambda, qd_dwt, Wt);
  updateParam(eta, regularizer, lambda, qd_dBias, B);

}

void update(const double eta,
	    const std::vector<MyMatrix> &gradB, 
	    const std::vector<MyMatrix> &A, 
	    const MyMatrix &X_batch,
	    const std::string regularizer, 
	    const double lambda,
	    std::vector<MyMatrix> &W,
	    std::vector<MyVector> &B,
	    std::vector<MyMatrix> &Mii,
	    std::vector<MyMatrix> &M0i,
	    std::vector<MyVector> &M00){
  
  const unsigned batch_size = X_batch.rows();

  for(unsigned i = 0; i < W.size(); i++){

    const MyMatrix* Atilde = &X_batch;
    if(i > 0){
      Atilde = &A[i-1];
    }

    const MyMatrix dw = (Atilde->transpose() * gradB[i])/batch_size;
    const MyVector gradB_avg = gradB[i].colwise().sum()/batch_size;
    
    MyMatrix qd_dw(dw.rows(), dw.cols());
    MyVector qd_dBias(gradB_avg.size());
    
    qdGradient(Mii[i], M0i[i], M00[i], dw, gradB_avg, qd_dw, qd_dBias);
    updateParam(eta, regularizer, lambda, qd_dw, W[i]);
    updateParam(eta, regularizer, lambda, qd_dBias, B[i]);
  }
}

void update(const double eta,
	    const std::vector<MyMatrix> &gradB,
	    const std::vector<MyMatrix> &A,
	    const MyMatrix &X_batch,
	    const std::string regularizer,
	    const double lambda,
	    MyMatrix &W_out,
	    std::vector<MySpMatrix> &W,
	    std::vector<MySpMatrix> &Wt,
	    std::vector<MyVector> &B,
	    MyMatrix &Mii_out,
	    MyMatrix &M0i_out,
	    MyVector &M00_out,
	    std::vector<MySpMatrix> &Mii,
	    std::vector<MySpMatrix> &M0i,
	    std::vector<MyVector> &M00){

  const unsigned batch_size = X_batch.rows();
  const MyMatrix* Atilde = &X_batch;

  // Update the hidden layers
  for(unsigned i = 0; i < W.size(); i++){
    opUpdateLayer(eta, batch_size, gradB[i], *Atilde, regularizer, lambda, Mii[i], M0i[i], M00[i], W[i], Wt[i], B[i]);
    Atilde = &A[i];
  }

  // Update the last layer
  const unsigned n_layers = W.size()+2;
  const MyMatrix dw_out = (Atilde->transpose() * gradB[n_layers-2])/batch_size;
  const MyVector gradB_avg_out = gradB[n_layers-2].colwise().sum()/batch_size;

  MyMatrix qd_dw_out(dw_out.rows(), dw_out.cols());
  MyVector qd_dBias_out(gradB_avg_out.size());

  qdGradient(Mii_out, M0i_out, M00_out, dw_out, gradB_avg_out, qd_dw_out, qd_dBias_out);
  updateParam(eta, regularizer, lambda, qd_dw_out, W_out);
  updateParam(eta, regularizer, lambda, qd_dBias_out, B[n_layers-2]);

}

void update(const double eta,
	    const std::vector<MyMatrix> &gradB, 
	    const std::vector<MyMatrix> &A, 
	    const MyMatrix &X_batch,
	    const std::string regularizer, 
	    const double lambda,
	    std::vector<MyMatrix> &W,
	    std::vector<MyVector> &B,
	    std::vector<MyMatrix> &Mii,
	    std::vector<MyVector> &M00){
  
  const unsigned batch_size = X_batch.rows();

  for(unsigned i = 0; i < W.size(); i++){

    const MyMatrix* Atilde = &X_batch;
    if(i > 0){
      Atilde = &A[i-1];
    }

    const MyMatrix dw = (Atilde->transpose() * gradB[i])/batch_size;
    const MyVector gradB_avg = gradB[i].colwise().sum()/batch_size;
    
    MyMatrix qd_dw(dw.rows(), dw.cols());
    MyVector qd_dBias(gradB_avg.size());
    
    diagGradient(Mii[i], M00[i], dw, gradB_avg, qd_dw, qd_dBias);
    updateParam(eta, regularizer, lambda, qd_dw, W[i]);
    updateParam(eta, regularizer, lambda, qd_dBias, B[i]);
  }
}


void updateTest(const double eta,
		const std::vector<MyMatrix> &gradB, 
		const std::vector<MyMatrix> &A, 
		const MyMatrix &X_batch,
		const std::string regularizer, 
		const double lambda,
		std::vector<MyMatrix> &W,
		std::vector<MyVector> &B,
		std::vector<MyMatrix> &Mii,
		std::vector<MyMatrix> &M0i,
		std::vector<MyVector> &M00){
  
  const unsigned batch_size = X_batch.rows(); 

  for(unsigned i = 0; i < W.size(); i++){

    const MyMatrix* Atilde = &X_batch;
    if(i > 0){
      Atilde = &A[i-1];
    }

    const MyMatrix dw = (Atilde->transpose() * gradB[i])/batch_size;
    const MyVector gradB_avg = gradB[i].colwise().sum()/batch_size;
    
    MyMatrix qd_dw(dw.rows(), dw.cols());
    MyVector qd_dBias(gradB_avg.size());
    
    qdGradient(Mii[i], M0i[i], M00[i], dw, gradB_avg, qd_dw, qd_dBias);

    updateParam(eta, regularizer, lambda, qd_dw, W[i]);
    updateParam(eta, regularizer, lambda, qd_dBias, B[i]);

    // Metric check
    for(unsigned j = 0; j < M00[i].size(); j++){
      assert(M00[i](j)>0.);
      for(unsigned k = 0; k < Mii[i].rows(); k++){
    	assert(M00[i](j) * Mii[i](k,j) > M0i[i](k,j) * M0i[i](k,j));
      }
    }
    
    MyMatrix dw_t = dw.transpose();
    MyVector dw_rav(Map<MyVector>(dw_t.data(), dw_t.cols()*dw_t.rows()));
    MyVector dw_rav2(gradB_avg.size() + dw_rav.size());
    dw_rav2 << gradB_avg, dw_rav;
    std::cout << "dw_rav " << dw_rav2.rows() << " " << dw_rav2.cols() << std::endl;

    
    MyMatrix qd_dw_t = qd_dw.transpose();
    MyVector qd_dw_rav(Map<MyVector>(qd_dw_t.data(), qd_dw_t.cols()*qd_dw_t.rows()));
    MyVector qd_dw_rav2(qd_dBias.size() + qd_dw_rav.size());
    qd_dw_rav2 << qd_dBias, qd_dw_rav;
    
    double test = dw_rav2.dot(qd_dw_rav2);
    std::cout << test << std::endl;
    assert(test>0);

  }
}

void computeLoss(const ActivationFunction &act_func,
		 const unsigned batch_size,
		 const MyMatrix &X,
		 const MyVector &Y,
		 const std::vector<ConvLayerParams> &conv_params,
		 const std::vector<PoolLayerParams> &pool_params,
		 const std::vector<MyMatrix> &conv_W,
		 const std::vector<MyVector> &conv_B,
		 const std::vector<MySpMatrix> &W,
		 const MyMatrix &W_out,
		 const std::vector<MyVector> &B,
		 double &loss,
		 double &accuracy){

  const MyMatrix* Atilde = &X;
  
  MyMatrix pool_z0;
  MyMatrix conv_A;
  for(unsigned i = 0; i < conv_W.size(); i++){
    
    MyMatrix conv_z0 = conv_W[i] * *Atilde;
    conv_z0.colwise() += conv_B[i];
    
    MyMatrix conv_act(conv_z0.rows(), conv_z0.cols());
    MyMatrix conv_ap(conv_z0.rows(), conv_z0.cols());
    
    act_func(conv_z0,conv_act,conv_ap);
    
    // Pool max for conv layer
    std::vector<unsigned> pool_idx_x;
    std::vector<unsigned> pool_idx_y;
    poolMax(batch_size, conv_params[i].N, pool_params[i].N, pool_params[i].Hf, pool_params[i].stride, conv_act, pool_z0, pool_idx_x, pool_idx_y);
    
    if(i < conv_W.size()-1){
      buildConvMatrix(batch_size, pool_params[i].N, conv_params[i+1].N, conv_params[i+1].Hf, conv_params[i+1].stride, conv_params[i+1].padding, pool_z0, conv_A);
      Atilde = &conv_A;
    }
  }
  
  // Convert the pool layer into a FNN layer
  MyMatrix z0;
  conv2Layer(batch_size, pool_params[conv_W.size()-1].N, pool_z0, z0);
  
  const unsigned n_layers = W.size() + 1;

  MyMatrix z = (z0 * W[0]).rowwise() + B[0].transpose();
  MyMatrix a(z.rows(), z.cols());    
  MyMatrix ap(z.rows(), z.cols());
  act_func(z, a, ap);

  for(unsigned i = 1; i < W.size()-1; i++){
    z = (a * W[i]).rowwise() + B[i].transpose();
    a.resize(z.rows(), z.cols());
    ap.resize(z.rows(), z.cols());

    act_func(z, a, ap); // TODO: replace without the computation of the derivative
  }

  a = (a*W_out).rowwise() + B[W.size()].transpose();
  
  MyMatrix out;
  softmax(a, out);
  
  double cum_loss = 0.;
  for(unsigned i = 0; i < out.rows(); i++){
    if(out(i,Y(i)) > 1e-6){
      cum_loss += -1. * log(out(i,Y(i)));
    }else{
      cum_loss += 15.0;
    }
  }

  // Compute the accuracy
  unsigned n_correct = 0;
  for(unsigned i = 0; i < out.rows(); i++){
    MyMatrix::Index idx;
    double val = out.row(i).maxCoeff(&idx);
    if(idx==Y(i))
      n_correct++;
  }
  // accuracy = (float)n_correct/out.rows();
  // loss = cum_loss/out.rows();
  accuracy = (float)n_correct;
  loss = cum_loss;
  
}


void evalModel(const ActivationFunction &eval_act_func,
	       const Params &params,
	       const unsigned n_batch,
	       const unsigned n_example,
	       const MyMatrix &X,
	       const MyVector &Y,
	       const std::vector<ConvLayerParams> &conv_params,
	       const std::vector<PoolLayerParams> &pool_params,
	       const std::vector<MyMatrix> &conv_W,
	       const std::vector<MyVector> &conv_B,
	       MyMatrix W_out,
	       std::vector<MySpMatrix> &W_eval,
	       std::vector<MyVector> &B,
	       double &acc_loss,
	       double &acc_accuracy){    

  acc_loss = 0.;
  acc_accuracy = 0.;
  
  // Training accuracy
  for(unsigned j = 0; j < n_batch; j++){
    
    // Mini-batch creation
    unsigned curr_batch_size = 0;
    MyMatrix X_batch;
    MyVector Y_batch;
    getMiniBatch(j, params.train_minibatch_size, X, Y, params, conv_params[0], curr_batch_size, X_batch, Y_batch);
    
    double loss = 0.;
    double accuracy = 0.;
    computeLoss(eval_act_func, curr_batch_size, X_batch, Y_batch, conv_params, pool_params, conv_W, conv_B, W_eval, W_out, B, loss, accuracy);

    acc_loss += loss;
    acc_accuracy += accuracy;
  }
  
  acc_loss/=n_example;
  acc_accuracy/=n_example;
}

// void evalModel(const ActivationFunction &eval_act_func,
// 	       const unsigned train_batch_size,
// 	       const MyMatrix &X_train,
// 	       const MyMatrix &Y_train,
// 	       const unsigned valid_batch_size,
// 	       const MyMatrix &X_valid,
// 	       const MyMatrix &Y_valid,
// 	       const std::vector<ConvLayerParams> &conv_params,
// 	       const std::vector<PoolLayerParams> &pool_params,
// 	       const std::vector<MyMatrix> &conv_W,
// 	       const std::vector<MyVector> &conv_B,
// 	       std::vector<MyMatrix> &W_eval,
// 	       std::vector<MyVector> &B,
// 	       double &train_loss,
// 	       double &train_accuracy,
// 	       double &valid_loss,
// 	       double &valid_accuracy){
    
//   // Training accuracy 
//   computeLoss(eval_act_func, train_batch_size, X_train, Y_train, conv_params, pool_params, conv_W, conv_B, W_eval, B, train_loss, train_accuracy);
  
//   // Validation accuracy
//   computeLoss(eval_act_func, valid_batch_size, X_valid, Y_valid, conv_params, pool_params, conv_W, conv_B, W_eval, B, valid_loss, valid_accuracy);

// }

// void adaptiveRule(const double train_loss, 
// 		  double &prev_loss, 
// 		  double &eta, 
// 		  std::vector<MyMatrix> &W, 
// 		  std::vector<MyVector> &B, 
// 		  std::vector<MyMatrix> &pW, 
// 		  std::vector<MyVector> &pB){
  
//   if(train_loss < prev_loss){
//     eta *= 1.2;
//     pW = W;
//     pB = B;
//     prev_loss = train_loss;
//   }else{
//     eta *= 0.5;
//     W = pW;
//     B = pB;
//   }
// }

void adaptiveRule(const double train_loss, 		 
		  double &prev_loss, 
		  double &eta,
		  std::vector<MyMatrix> &W, 
		  std::vector<MyVector> &B,
		  std::vector<MyMatrix> &pMii,
		  std::vector<MyMatrix> &pM0i,
		  std::vector<MyVector> &pM00,
		  std::vector<MyMatrix> &pW, 
		  std::vector<MyVector> &pB,
		  std::vector<MyMatrix> &ppMii,
		  std::vector<MyMatrix> &ppM0i,
		  std::vector<MyVector> &ppM00){
  
  if(train_loss < prev_loss){
    eta *= 1.2;
    pW = W;
    pB = B;

    ppMii = pMii;
    ppM0i = pM0i;
    ppM00 = pM00;
    
    prev_loss = train_loss;
  }else{
    eta *= 0.5;
    
    W = pW;
    B = pB;

    pMii = ppMii;
    pM0i = ppM0i;
    pM00 = ppM00;    
  }
}
