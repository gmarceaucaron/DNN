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
 * \file nn_ops.hpp
 * \brief Implementation of the functions required for neural networks 
 * \author Gaetan Marceau Caron & Yann Ollivier
 * \version 1.0
 */

#include <functional>
#include <Eigen/Dense>
#include <algorithm>
#include <assert.h>
#include <limits> 

#include "utils.hpp"

/**
 *  \brief Initialize the weights of a layer with a reweighted normal noise
 *
 *  @param n_act0 number of input units
 *  @param n_act1 number of output units
 *  @param sigma stddev of the normal noise
 *  @param W weight matrix returned by the function
 */
void initWeight(const unsigned n_act0,
				const unsigned n_act1,
				const double sigma,
				MyMatrix& W){

	// Declare the normalized normal random variable 
	normal_distribution<double> normal(0,sigma/sqrt(n_act0));

	// Initialize the weight matrix
	W.resize(n_act0,n_act1);

	// Generate randomly the weights
	for(unsigned i = 0; i < n_act1; i++){
		for(unsigned j = 0; j < n_act0; j++){
			W(j,i) = normal(gen);
		}
	}
}

/**
 *  \brief Initialize the parameters of the Neural Network computational graph 
 *
 *  @param nn_arch number of activation units for each layer
 *  @param act_func string representing the activation function
 *  @param W a standard vector of weight matrices (one per layer)
 *  @param B a standard vector of bias vectors (one per layer)
 *  @return the number of parameters
 */
int initNetwork(const std::vector<unsigned> &nn_arch,
				const std::string act_func, 
				std::vector<MyMatrix> &W,
				std::vector<MyVector> &B){
  
	// Determine the noise amplitude for the initialization of the weight matrix
	const unsigned n_layers = nn_arch.size();
	double sigma = 1.0;
	if(act_func=="sigmoid"){
		sigma = 4.0;
	}
	unsigned param_counter = 0;
  
	// Initialize the weights according to the chosen activation function
	for(unsigned i = 0; i < n_layers-1; i++){
		initWeight(nn_arch[i], nn_arch[i+1], sigma, W[i]);
		if(act_func=="sigmoid"){
			B[i] = -0.5 * W[i].colwise().sum();
		}else if(act_func=="tanh"){
			B[i] = MyVector::Zero(W[i].cols());
		}else if(act_func=="relu"){ //@TODO: find the optimal way to initialize relu
			B[i] = MyVector::Zero(W[i].cols());
		}else{
			std::cout << "Not implemented!" << std::endl;
			assert(false);
		}
		param_counter += nn_arch[i] * nn_arch[i+1] + nn_arch[i+1];
	}
	return param_counter;
}

/**
 *  \brief Implementation of the sign function 
 *
 *  @param x an input value
 *  @return the sign of x
 */
double signFunc(double x){
	if(x<0.0)
		return -1.0;
	else if(x>0.0)
		return 1.0;
	else
		return 0.0;
}

/**
 *  \brief Implementation of the square function 
 *
 *  @param x the pre-activation value
 *  @return the square of x
 */
double squareFunc(double x){
	return x*x;
}

/**
 *  \brief Implementation of the square-root function 
 *
 *  @param x the pre-activation value
 *  @return the square-root of x
 */
double sqrtFunc(double x){
	return sqrt(x);
}

/**
 *  \brief Implementation of the translation function 
 *
 *  @param x the pre-activation value
 *  @return x plus a constant
 */
double sumCstFunc(double x, double cst){
	return x+cst;
}

/**
 *  \brief Implementation of the softmax function
 *  Take a matrix (n_example X n_class) of values and
 *  return a matrix (n_example X n_class) where each line
 *  is a probability distribution over the classes
 *
 *  @param a the pre-activation values
 *  @param out the probability distribution
 */
void softmax(const MyMatrix &a, 
			 MyMatrix& out){

	// Translate the pre-activation for numerical stability
	MyVector max_coeff = a.rowwise().maxCoeff();
	MyMatrix centered_a = a.colwise() - max_coeff;
  
	// Compute the exponential of the entries
	for(unsigned j = 0; j < a.cols(); j++){ // Optimization for column-major
		for(unsigned i = 0; i < a.rows(); i++){  
			centered_a(i,j) = exp(centered_a(i,j));
		}
	}

	// Normalize
	MyVector sum_coeff = centered_a.rowwise().sum();
	out = centered_a.array().colwise() / sum_coeff.array();
}

///////////////////////////////////////////////////
//Activation functions
///////////////////////////////////////////////////

/**
 *  \brief Implementation of the logistic function and its derivative
 *
 *  @param deriv_flag a flag for computing the derivative at the same time
 *  @param z the pre-activation matrix (n_example X n_unit)
 *  @param a the activation matrix (n_example X n_unit)
 *  @param da the derivative activation matrix (n_example X n_unit)
 */
void logistic(const bool deriv_flag,
			  const MyMatrix &z, 
			  MyMatrix &a, 
			  MyMatrix &da){
	if(deriv_flag){
		for(int j = 0; j < z.cols(); j++){ // Optimization for column-major
			for(int i = 0; i < z.rows(); i++){
				a(i,j) = 1.0/(1.0+exp(-z(i,j)));
				da(i,j) = a(i,j) * (1.0-a(i,j));
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

/**
 *  \brief Implementation of the tanh function and its derivative
 *
 *  @param deriv_flag a flag for computing the derivative at the same time
 *  @param z the pre-activation matrix (n_example X n_unit)
 *  @param a the activation matrix (n_example X n_unit)
 *  @param da the derivative activation matrix (n_example X n_unit)
 */
void my_tanh(const bool deriv_flag,
			 const MyMatrix &z, 
			 MyMatrix &a, 
			 MyMatrix &da){
	if(deriv_flag){
		for(int j = 0; j < z.cols(); j++){ // Optimization for column-major
			for(int i = 0; i < z.rows(); i++){
				a(i,j) = tanh(z(i,j));
				da(i,j) = (1.0-a(i,j)*a(i,j));
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

/**
 *  \brief Implementation of the rectified linear unit (ReLU) function and its derivative
 *
 *  @param deriv_flag a flag for computing the derivative at the same time
 *  @param z the pre-activation matrix (n_example X n_unit)
 *  @param a the activation matrix (n_example X n_unit)
 *  @param da the derivative activation matrix (n_example X n_unit)
 */
void relu(const bool deriv_flag,
		  const MyMatrix &z, 
		  MyMatrix &a, 
		  MyMatrix &da){
	if(deriv_flag){
		for(int j = 0; j < z.cols(); j++){ // Optimization for column-major
			for(int i = 0; i < z.rows(); i++){
				a(i,j) = max(0.0,z(i,j));
				if(z(i,j)>0)
					da(i,j) = 1.0;
				else
					da(i,j) = 0.0;
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

///////////////////////////////////////////////////
//Dropout Regularization
///////////////////////////////////////////////////

/**
 *  \brief Implementation of the dropout regularization technique
 *
 *  @param prob the probability of dropout
 *  @param A a matrix to regularize (n_example X n_unit)
 *  @param B another matrix to regularize with the same mask (n_example X n_unit)
 */
void dropout(const double prob,
			 MyMatrix &A,
			 MyMatrix &B){

	// The mask is a realization of (n_example X n_unit) realizations of i.i.d bernoulli trials
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

/**
 *  \brief Implementation of the dropout regularization technique
 *
 *  @param prob the probability of dropout
 *  @param A a matrix to regularize (n_example X n_unit)
 */
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

///////////////////////////////////////////////////
//Standard Neural Network Learning
///////////////////////////////////////////////////

/**
 *  \brief Implementation of the forward propagation algorithm
 *
 *  @param dropout_flag the flag for activating dropout regularization
 *  @param dropout_prob the probability of dropout (dropout_flag must be true)
 *  @param act_func the reference of the activation function
 *  @param W a standard vector of weight matrices (one per layer)
 *  @param B a standard vector of bias vectors (one per layer)
 *  @param X_batch a matrix containing the training examples of the current minibatch
 *  @param Z a standard vector of pre-activation values (one per layer)
 *  @param A a standard vector of activation values (one per layer)
 *  @param dA a standard vector of the derivatives of the activation values (one per layer)
 */
void fprop(const bool dropout_flag,
		   const double dropout_prob,
		   const ActivationFunction &act_func,
		   const std::vector<MyMatrix> &W, 
		   const std::vector<MyVector> &B,
		   const MyMatrix &X_batch,
		   std::vector<MyMatrix> &Z,
		   std::vector<MyMatrix> &A,
		   std::vector<MyMatrix> &dA){

	// Verify if the dropout_flag is activated, if so, then we need
	// to copy the X_batch in order to do dropout on it
	const MyMatrix* Atilde = &X_batch;
	if(dropout_flag){
		Atilde = new MyMatrix(X_batch);
		dropout(0.8, *const_cast<MyMatrix *>(Atilde));
	}

	// For each layer, forward propagate the data and compute the activations (also perform dropout if flag is set)
	for(unsigned i = 0; i < A.size(); i++){
		Z[i] = (*Atilde * W[i]).rowwise() + B[i].transpose();
		A[i].resize(Z[i].rows(), Z[i].cols());
		dA[i].resize(Z[i].rows(), Z[i].cols());
		act_func(Z[i],A[i],dA[i]);
		if(dropout_flag){
			dropout(dropout_prob, A[i], dA[i]);
			if(i==0)
				delete Atilde;
		}
		Atilde = &A[i];
	}

	// For the output layer, there is no activation (softmax is done later)
	Z[A.size()] = (A[A.size()-1] * W[A.size()]).rowwise() + B[A.size()].transpose();

}

/**
 *  \brief Implementation of the backpropagation algorithm
 *
 *  @param W a standard vector of weight matrices (one per layer)
 *  @param dA a standard vector of the derivatives of the activation values (one per layer)
 *  @param gradB a standard vector of the gradient w.r.t. the pre-activation values (one per layer)
 */
void bprop(const std::vector<MyMatrix> &W,
		   const std::vector<MyMatrix> &dA,
		   std::vector<MyMatrix> &gradB){
  
	const unsigned n_layers = W.size() + 1;
	for(unsigned i = 0; i < W.size()-1; i++){
		const unsigned rev_i = n_layers - i - 3;
		gradB[rev_i] = (gradB[rev_i+1] * W[rev_i+1].transpose()).cwiseProduct(dA[rev_i]);
	}
}

/**
 *  \brief Templated function for updating matrices or vectors 
 *
 *  @param eta the gradient descent step-size
 *  @param regularizer the string of the norm regularizer (L1 or L2)
 *  @param lambda the amplitude of the regularization term
 *  @param dparams a matrix/vector containing the updates
 *  @param params a matrix/vector to be updated
 */
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

/**
 *  \brief Implementation of the update function of the gradient descent algorithm 
 *
 *  @param eta the gradient descent step-size
 *  @param gradB a standard vector of the gradient w.r.t. the pre-activation values (one per layer)
 *  @param A a standard vector of activation values (one per layer)
 *  @param X_batch a matrix containing the training examples of the current minibatch
 *  @param regularizer the string of the norm regularizer (L1 or L2)
 *  @param lambda the amplitude of the regularization term
 *  @param W a standard vector of weight matrices (one per layer)
 *  @param B a standard vector of bias vectors (one per layer)
 */
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

		// Compute the update of the weights
		MyMatrix dw = (Atilde->transpose() * gradB[i])/batch_size;
		updateParam(eta, regularizer, lambda, dw, W[i]);

		// Compute the update of the bias
		const MyVector gradB_avg = gradB[i].colwise().sum()/batch_size;
		updateParam(eta, regularizer, lambda, gradB_avg, B[i]);
	}
}


/**
 *  \brief function that performs the adagrad update rule
 *
 *  @param eta the gradient descent step-size
 *  @param gradB a standard vector of the gradient w.r.t. the pre-activation values (one per layer)
 *  @param A a standard vector of activation values (one per layer)
 *  @param X_batch a matrix containing the training examples of the current minibatch
 *  @param regularizer the string of the norm regularizer (L1 or L2)
 *  @param lambda the amplitude of the regularization term
 *  @param mat_reg a numerical regularization term s.t. there is no division by zero
 *  @param autocorr the autocorrelation coefficient of the adagrad update rule
 *  @param W a standard vector of weight matrices (one per layer)
 *  @param B a standard vector of bias vectors (one per layer)
 *  @param mu_dW a standard vector of the rolling averages of the weight matrices (one per layer)
 *  @param mu_dB a standard vector of the rolling averages of the bias vectors (one per layer)
 */
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

		// Compute the update of the weights
		const MyMatrix dw = (Atilde->transpose() * gradB[i])/batch_size;

		// Initialize the rolling average or update it
		if(ada_first || autocorr<1e-6){
			mu_dW[i] = dw.array().square();
		}else{
			mu_dW[i] = autocorr * mu_dW[i].array() + (1.0-autocorr) * dw.array().square();
		}

		// Compute the adagrad update rule
		MyMatrix temp = mu_dW[i].cwiseSqrt().array() + mat_reg;
		MyMatrix ada_dw = dw.cwiseQuotient(temp);

		// Update the weight parameters
		updateParam(eta, regularizer, lambda, ada_dw, W[i]);

		// Compute the update of the weights
		const MyVector gradB_avg = gradB[i].colwise().sum() / batch_size;

		// Initialize the rolling average or update it
		if(ada_first || autocorr<1e-6){
			mu_dB[i] = gradB_avg.array().square();
		}else{
			mu_dB[i] = autocorr * mu_dB[i].array() + (1.0-autocorr) * gradB_avg.array().square();
		}

		// Compute the adagrad update rule
		MyVector temp2 = mu_dB[i].cwiseSqrt().array() + mat_reg;
		MyVector ada_dB = gradB_avg.cwiseQuotient(temp2);

		// Update the bias parameters
		updateParam(eta, regularizer, lambda, ada_dB, B[i]);
	}
}

/**
 *  \brief function that compute the Negative Log-likelihood and the accuracy of the model
 *
 *  @param act_func the reference of the activation function
 *  @param X a matrix containing the training examples
 *  @param Y a vector containing the labels of the examples
 *  @param W a standard vector of weight matrices (one per layer)
 *  @param B a standard vector of bias vectors (one per layer)
 *  @param loss the negative log-likelihood
 *  @param accuracy the accuracy 
 */
void computeLoss(const ActivationFunction &act_func,
				 const MyMatrix &X,
				 const MyVector &Y,
				 const std::vector<MyMatrix> &W, 
				 const std::vector<MyVector> &B,
				 double &loss,
				 double &accuracy){

	// Perform the forward propagation algorithm without stocking the intermediate values
	MyMatrix z = (X * W[0]).rowwise() + B[0].transpose();
	MyMatrix a(z.rows(), z.cols());    
	MyMatrix ap(z.rows(), z.cols());
	act_func(z, a, ap);

	for(unsigned i = 1; i < W.size()-1; i++){
		z = (a * W[i]).rowwise() + B[i].transpose();
		a.resize(z.rows(), z.cols());
		ap.resize(z.rows(), z.cols());

		act_func(z, a, ap); // TODO: replace without the computation of the derivative
	}
  
	a = (a*W[W.size()-1]).rowwise() + B[W.size()-1].transpose();

	// Compute the output of the network
	MyMatrix out;
	softmax(a, out);

	// Compute the cumulative negative log-likelihood
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

	// Compute the averages
	accuracy = (float)n_correct/out.rows();
	loss = cum_loss/out.rows();
}

/**
 *  \brief function that compute the Negative Log-likelihood and the accuracy of the model
 *         (cannot be used with dropout since the mask will be activated!)
 *
 *  @param act_func the reference of the activation function
 *  @param X a matrix containing the training examples
 *  @param Y a vector containing the labels of the examples
 *  @param W a standard vector of weight matrices (one per layer)
 *  @param B a standard vector of bias vectors (one per layer)
 *  @param params the parameters of the program
 *  @param loss the negative log-likelihood
 *  @param accuracy the accuracy 
 */
void computeLoss(const ActivationFunction &act_func,
				 const MyMatrix &X,
				 const MyVector &Y,
				 const std::vector<MyMatrix> &W, 
				 const std::vector<MyVector> &B,
				 const Params params,
				 double &loss,
				 double &accuracy){

	const unsigned n_layers = params.nn_arch.size();
  
	// Forward propagation
	std::vector<MyMatrix> Z(n_layers-1);
	std::vector<MyMatrix> A(n_layers-2);
	std::vector<MyMatrix> dA(n_layers-2);
	fprop(params.dropout_flag, params.dropout_prob, act_func, W, B, X, Z, A, dA);
    
	// Compute the output and the error
	MyMatrix out;
	softmax(Z[n_layers-2], out);
    
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
	accuracy = (float)n_correct/out.rows();
	loss = cum_loss/out.rows();
}

/**
 *  \brief function that evaluate the model on the training dataset and the validation dataset
 *
 *  @param eval_act_func the reference of the activation function
 *  @param X_train a matrix containing the training examples
 *  @param Y_train a vector containing the labels of the training examples
 *  @param X_valid a matrix containing the validation examples
 *  @param Y_valid a vector containing the labels of the validation examples
 *  @param params the parameters of the program
 *  @param W_eval a copy of the standard vector of weight matrices (one per layer)
 *  @param B a standard vector of bias vectors (one per layer)
 *  @param train_loss the negative log-likelihood on the training set
 *  @param train_accuracy the accuracy on the training set 
 *  @param valid_loss the negative log-likelihood on the validation set
 *  @param valid_accuracy the accuracy on the validation set 
 */
void evalModel(const ActivationFunction &eval_act_func,
			   const MyMatrix &X_train,
			   const MyMatrix &Y_train,
			   const MyMatrix &X_valid,
			   const MyMatrix &Y_valid,
			   const Params &params,
			   std::vector<MyMatrix> W_eval,
			   std::vector<MyVector> B,
			   double &train_loss,
			   double &train_accuracy,
			   double &valid_loss,
			   double &valid_accuracy){
  
	if(params.dropout_flag && !params.dropout_eval){

		// For dropout, the weights must be rescaled according to the probability of occurence
		W_eval[0] *= 0.8;
		for(unsigned k = 1; k < W_eval.size(); k++){
			W_eval[k] *= params.dropout_prob;
		}

		// Training accuracy 
		computeLoss(eval_act_func, X_train, Y_train, W_eval, B, train_loss, train_accuracy);
    
		// Validation accuracy
		computeLoss(eval_act_func, X_valid, Y_valid, W_eval, B, valid_loss, valid_accuracy);

	}else{
		// Training accuracy 
		computeLoss(eval_act_func, X_train, Y_train, W_eval, B, params, train_loss, train_accuracy);
    
		// Validation accuracy
		computeLoss(eval_act_func, X_valid, Y_valid, W_eval, B, params, valid_loss, valid_accuracy);
	}
}

/**
 *  \brief simple implementation of a decreasing step-size for ensuring convergence
 *
 *  @param t number of iterations
 *  @param init_eta initial step-size
 *  @param eta adjusted step-size
 */
void updateStepsize(const unsigned t,
					const double init_eta,
					double &eta){
	eta = init_eta/sqrt(t);
}

///////////////////////////////////////////////////
//Riemannian Metric Neural Network
///////////////////////////////////////////////////

/**
 *  \brief Function that builds a diagonal metric from the gradient w.r.t. the pre-activation
 *
 *  @param gradB_sq a standard vector of the square of the gradient w.r.t. the pre-activation values (one per layer)
 *  @param A a standard vector of activation values (one per layer)
 *  @param X_batch a matrix containing the training examples of the current minibatch
 *  @param W a standard vector of weight matrices (one per layer)
 *  @param mat_reg a numerical regularization term s.t. the metric is invertible
 *  @param Mii a standard vector of the diagonals of the block sub-matrices of the Riemannian metric (one per layer)
 *  @param M00 a standard vector of the bias of the block sub-matrices of the Riemannian metric (one per layer)
 */
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

/**
 *  \brief Function that builds a quasi-diagonal metric from the gradient w.r.t. the pre-activation
 *
 *  @param gradB_sq a standard vector of the square of the gradient w.r.t. the pre-activation values (one per layer)
 *  @param A a standard vector of activation values (one per layer)
 *  @param X_batch a matrix containing the training examples of the current minibatch
 *  @param W a standard vector of weight matrices (one per layer)
 *  @param mat_reg a numerical regularization term s.t. the metric is invertible
 *  @param Mii a standard vector of the diagonals of the block sub-matrices of the Riemannian metric (one per layer)
 *  @param Mi0 a standard vector of the weights time the bias (first line) of the block sub-matrices of the Riemannian metric (one per layer)
 *  @param M00 a standard vector of the bias of the block sub-matrices of the Riemannian metric (one per layer)
 */
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

/**
 *  \brief Update the quasi-diagonal approximation of the metric with an exponential moving average
 *
 *  @param init_flag a flag for initializing the exponential moving average
 *  @param gamma the coefficient of the exponential moving average
 *  @param Mii a standard vector of the diagonals of the block sub-matrices of the Riemannian metric (one per layer)
 *  @param Mi0 a standard vector of the weights time the bias (first line) of the block sub-matrices of the Riemannian metric (one per layer)
 *  @param M00 a standard vector of the bias of the block sub-matrices of the Riemannian metric (one per layer)
 *  @param pMii a standard vector of the previous exponential moving average of Mii 
 *  @param pMi0 a standard vector of the previous exponential moving average of M0i 
 *  @param pM00 a standard vector of the previous exponential moving average of M00
 */
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

/**
 *  \brief Update the diagonal approximation of the metric with an exponential moving average
 *
 *  @param init_flag a flag for initializing the exponential moving average
 *  @param gamma the coefficient of the exponential moving average
 *  @param Mii a standard vector of the diagonals of the block sub-matrices of the Riemannian metric (one per layer)
 *  @param M00 a standard vector of the bias of the block sub-matrices of the Riemannian metric (one per layer)
 *  @param pMii a standard vector of the previous exponential moving average of Mii 
 *  @param pM00 a standard vector of the previous exponential moving average of M00
 */
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

/**
 *  \brief Compute the natural gradient with the quasi-diagonal approximation
 *
 *  @param Mii the diagonals of the block sub-matrices of the Riemannian metric (one per layer)
 *  @param Mi0 the weights time the bias (first line) of the block sub-matrices of the Riemannian metric (one per layer)
 *  @param M00 the bias of the block sub-matrices of the Riemannian metric (one per layer)
 *  @param dW the gradient w.r.t. the weights
 *  @param dB the gradient w.r.t. the bias
 *  @param qd_dW the quasi-diagonal natural gradient w.r.t. the weights
 *  @param qd_dB the quasi-diagonal natural gradient w.r.t. the bias
 */
void qdGradient(const MyMatrix &Mii,
				const MyMatrix &M0i,
				const MyVector &M00,
				const MyMatrix &dW,
				const MyVector &dB,
				MyMatrix &qd_dW,
				MyVector &qd_dB){

	double temp = 0.0;
	for(unsigned k = 0; k < dW.cols(); k++){ 
		for(unsigned i = 0; i < dW.rows(); i++){
			qd_dW(i,k) = (dW(i,k) * M00(k) - dB(k) * M0i(i,k)) / ((Mii(i,k) * M00(k)) - (M0i(i,k) * M0i(i,k)));
			temp += (M0i(i,k) / M00(k)) * qd_dW(i,k);
		}
		qd_dB(k) = (dB(k) / M00(k)) - temp;
		temp = 0.0;
	}
}

/**
 *  \brief Compute the natural gradient with the diagonal approximation
 *
 *  @param Mii the diagonals of the block sub-matrices of the Riemannian metric (one per layer)
 *  @param M00 the bias of the block sub-matrices of the Riemannian metric (one per layer)
 *  @param dW the gradient w.r.t. the weights
 *  @param dB the gradient w.r.t. the bias
 *  @param qd_dW the quasi-diagonal natural gradient w.r.t. the weights
 *  @param qd_dB the quasi-diagonal natural gradient w.r.t. the bias
 */
void diagGradient(const MyMatrix &Mii,
				  const MyVector &M00,
				  const MyMatrix &dW,
				  const MyVector &dB,
				  MyMatrix &qd_dW,
				  MyVector &qd_dB){

	for(unsigned k = 0; k < dW.cols(); k++){ 
		for(unsigned i = 0; i < dW.rows(); i++){
			qd_dW(i,k) = dW(i,k)/Mii(i,k);
		}
		qd_dB(k) = dB(k)/M00(k);
	}
}

/**
 *  \brief Implementation of the update function of the quasi-diagonal natural gradient descent algorithm 
 *
 *  @param eta the gradient descent step-size
 *  @param gradB a standard vector of the gradient w.r.t. the pre-activation values (one per layer)
 *  @param A a standard vector of activation values (one per layer)
 *  @param X_batch a matrix containing the training examples of the current minibatch
 *  @param regularizer the string of the norm regularizer (L1 or L2)
 *  @param lambda the amplitude of the regularization term
 *  @param W a standard vector of weight matrices (one per layer)
 *  @param B a standard vector of bias vectors (one per layer)
 *  @param Mii the diagonals of the block sub-matrices of the Riemannian metric (one per layer)
 *  @param Mi0 the weights time the bias (first line) of the block sub-matrices of the Riemannian metric (one per layer)
 *  @param M00 the bias of the block sub-matrices of the Riemannian metric (one per layer)
 */
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

		// Compute the update of the weights
		const MyMatrix dw = (Atilde->transpose() * gradB[i])/batch_size;
		MyMatrix qd_dw(dw.rows(), dw.cols());

		// Compute the update of the bias
		const MyVector gradB_avg = gradB[i].colwise().sum()/batch_size;
		MyVector qd_dBias(gradB_avg.size());

		// Compute the natural gradient
		qdGradient(Mii[i], M0i[i], M00[i], dw, gradB_avg, qd_dw, qd_dBias);

		// Update the parameters
		updateParam(eta, regularizer, lambda, qd_dw, W[i]);
		updateParam(eta, regularizer, lambda, qd_dBias, B[i]);
	}
}

/**
 *  \brief Implementation of the update function of the diagonal natural gradient descent algorithm 
 *
 *  @param eta the gradient descent step-size
 *  @param gradB a standard vector of the gradient w.r.t. the pre-activation values (one per layer)
 *  @param A a standard vector of activation values (one per layer)
 *  @param X_batch a matrix containing the training examples of the current minibatch
 *  @param regularizer the string of the norm regularizer (L1 or L2)
 *  @param lambda the amplitude of the regularization term
 *  @param W a standard vector of weight matrices (one per layer)
 *  @param B a standard vector of bias vectors (one per layer)
 *  @param Mii the diagonals of the block sub-matrices of the Riemannian metric (one per layer)
 *  @param M00 the bias of the block sub-matrices of the Riemannian metric (one per layer)
 */
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

/**
 *  \brief Implementation of a simple 0.5/1.2 step-size adaptation rule
 *
 *  @param train_loss the negative log-likelihood associated to the current parameters
 *  @param prev_loss the negative log-likelihood associated to the previous parameters
 *  @param eta the current step-size
 *  @param W a standard vector of weight matrices (one per layer)
 *  @param B a standard vector of bias vectors (one per layer)
 *  @param pMii a standard vector of the exponential moving average of Mii 
 *  @param pMi0 a standard vector of the exponential moving average of M0i 
 *  @param pM00 a standard vector of the exponential moving average of M00
 *  @param pW a standard vector of weight matrices of the previous iteration (one per layer)
 *  @param pB a standard vector of bias vectors of the previous iteration (one per layer)
 *  @param ppMii a standard vector of the exponential moving average of Mii of the previous iteration 
 *  @param ppMi0 a standard vector of the exponential moving average of M0i of the previous iteration
 *  @param ppM00 a standard vector of the exponential moving average of M00 of the previous iteration
 */
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

/**
 *  \brief Implementation of the metric backpropagation algorithm
 *
 *  @param W a standard vector of weight matrices (one per layer)
 *  @param dA a standard vector of the derivatives of the activation values (one per layer)
 *  @param bp_gradB a standard vector of the backpropagated metric (one per layer)
 */
void qdBpmBprop(const std::vector<MyMatrix> &W,
				const std::vector<MyMatrix> &dA,
				std::vector<MyMatrix> &bp_gradB){
  
	const unsigned n_layers = W.size() + 1;
	for(unsigned i = 0; i < W.size()-1; i++){
		const unsigned rev_i = n_layers - i - 3;
		MyMatrix W_sq = W[rev_i+1].unaryExpr(std::ptr_fun(squareFunc));
		bp_gradB[rev_i] = (bp_gradB[rev_i+1] * W_sq.transpose()).cwiseProduct(dA[rev_i].array().square().matrix());
	}
}

/**
 *  \brief Function that sample a label randomly according to the output distribution and evaluate the gradient w.r.t. to the output
 *         This function is required by the algorithm qdMCNat
 *
 *  @param out the probability distribution over the output (values returned by softmax)
 *  @param mc_error the gradient w.r.t. to the output
 */
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


////////////////////////////////////////////////////////////////////
// Useless functions (intended for testing ideas that did not work)
////////////////////////////////////////////////////////////////////
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

void computeISError(const MyMatrix &out,
					const MyMatrix &one_hot_batch,
					const double q, 
					MyMatrix &mc_error){

	const unsigned n_labels = out.cols();
  
	mc_error.setZero(out.rows(),out.cols());
	const double true_prob = q + (1.0 - q)/n_labels;
  
	for(unsigned i = 0; i < out.rows(); i++){
		std::vector<double> probs(out.cols());
		for(unsigned j = 0; j < out.cols(); j++){
			mc_error(i,j) = out(i,j);
			if(one_hot_batch(i,j)==1){
				probs[j] =  true_prob;
			}
			else{
				probs[j] =  (1.0-q)/n_labels;
			}
		}
		std::discrete_distribution<int> dist(probs.begin(), probs.end());
    
		const unsigned label = dist(gen);
		for(unsigned j = 0; j < out.cols(); j++){
			if(j==label){
				mc_error(i,j) = (out(i,j)-1.0) * out(i,j)/probs[label];
			}
			else{
				if(probs[j] < 1e-16)
					mc_error(i,j) = out(i,j) * out(i,j);
				else
					mc_error(i,j) = out(i,j) * out(i,j)/probs[j];
			}
		}
	}
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

////////////////////////////////////////////////////////////////////
// Function adapted for testing purposes
////////////////////////////////////////////////////////////////////
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
