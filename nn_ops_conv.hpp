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
 * \file nn_ops_conv.hpp
 * \brief Implementation of the functions required for convolutional neural networks 
 * \author Gaetan Marceau Caron & Yann Ollivier
 * \version 1.0
 */

#include "utils.hpp"
#include "utils_conv.hpp"
#include "nn_ops.hpp"

/**
 *  \brief Transpose the convolution weights according to the characteristic of the filter
 *
 *  @param conv_W a weight matrix of a convolutional layer
 *  @param n_chan the number of channels 
 *  @param Hf the size of the filter
 *  @param conv_W_T the transposed weight matrix of a convolutional layer
 */
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

/**
 *  \brief Initialize the weights of a convolutional layer with a small normal noise (0.01)
 *
 *  @param conv_params a standard vector with parameters for each convolutional layer
 *  @param convW a standard vector with weight matrix for each convolutional layer
 *  @param convW_T a standard vector with transposed weight matrix for each convolutional layer
 *  @param convB a standard vector with bias vector for each convolutional layer
 */
unsigned initConvLayer(const std::vector<ConvLayerParams> &conv_params,
					   std::vector<MyMatrix> &convW,
					   std::vector<MyMatrix> &convW_T,
					   std::vector<MyVector> &convB){
	
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0,0.01); //@TODO parametrize the noise amplitude

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
	return n_params;
}

/**
 *  \brief Initialize the parameters of the convolutional Neural Network computational graph 
 *
 *  @param nn_arch number of activation units for each layer
 *  @param act_func string representing the activation function
 *  @param conv_params a standard vector with parameters for each convolutional layer
 *  @param pool_params a standard vector with parameters for each pooling layer
 *  @param W a standard vector of weight matrices (one per layer)
 *  @param B a standard vector of bias vectors (one per layer)
 *  @param convW a standard vector of weight matrices (one per layer)
 *  @param convW_T a standard vector of bias vectors (one per layer)
 *  @param convB a standard vector of weight matrices (one per layer)
 *  @return the number of parameters
 */
int initNetwork(const std::vector<unsigned> &nn_arch, 
				const std::string act_func, 
				const std::vector<ConvLayerParams> &conv_params,
				const std::vector<PoolLayerParams> &pool_params,
				std::vector<MyMatrix> &W,
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
  
	// Initialize the weights
	for(unsigned i = 0; i < n_layers-1; i++){
		initWeight(nn_arch[i], nn_arch[i+1], sigma, W[i]);
		if(act_func=="sigmoid"){
			B[i] = -0.5 * W[i].colwise().sum();
		}else if(act_func=="tanh"){
			B[i] = MyVector::Zero(W[i].cols());
		}else if(act_func=="relu"){ // TODO: find the right way to initialize relu
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
 *  \brief Perform a max pooling operation 
 *
 *  @param n_img the number of images
 *  @param conv_N1
 *  @param conv_N2
 *  @param F
 *  @param S
 *  @param conv_layer
 *  @param pool_layer
 *  @param pool_idx_x
 *  @param pool_idx_y
 */
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


/**
 *  \brief Implementation of the forward propagation algorithm for convolutional layer
 *
 *  @param batch_size number of examples in the batch
 *  @param conv_params a standard vector with parameters for each convolutional layer
 *  @param pool_params a standard vector with parameters for each pooling layer
 *  @param act_func string representing the activation function
 *  @param conv_W
 *  @param conv_B
 *  @param X_batch a matrix containing the training examples of the current minibatch
 *  @param conv_A
 *  @param conv_dA
 *  @param z0 
 *  @param poolIdxX
 *  @param poolIdxY
 */
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

/**
 *  \brief Implementation of the backpropagation algorithm for convolutional layer
 *
 *  @param batch_size number of examples in the batch
 *  @param conv_params a standard vector with parameters for each convolutional layer
 *  @param pool_params a standard vector with parameters for each pooling layer
 *  @param convW_T a standard vector of bias vectors (one per layer)
 *  @param convAp
 *  @param pool_gradB
 *  @param gradB
 *  @param poolIdxX
 *  @param poolIdxY
 */
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

		double prev_time = gettime();
		pool2conv(batch_size, conv_params[rev_l].n_filter, conv_params[rev_l].N, pool_params[rev_l].N, poolIdxX[rev_l], poolIdxY[rev_l], *const_cast<MyMatrix *>(Atilde), conv_gradB_act);

		prev_time = gettime();
		gradB[rev_l] = conv_gradB_act.cwiseProduct(convAp[rev_l]);
    
		if(rev_l>0){
			MyMatrix conv_mat_pool;

			prev_time = gettime();
			buildConvMatrix(batch_size, conv_params[rev_l].N, pool_params[rev_l-1].N, conv_params[rev_l].Hf, conv_params[rev_l].stride, pool_params[rev_l-1].N-conv_params[rev_l].N, gradB[rev_l], conv_mat_pool);

			*const_cast<MyMatrix *>(Atilde) = convW_T[rev_l] * conv_mat_pool;
		}
	}
}

/**
 *  \brief Implementation of the update function of the convolution layers (required for testing) 
 *
 *  @param eta the gradient descent step-size
 *  @param conv_gradB a standard vector of the gradient w.r.t. the pre-activation values (one per conv layer)
 *  @param conv_A a standard vector of activation values (one per conv layer)
 *  @param X_batch a matrix containing the training examples of the current minibatch
 *  @param regularizer the string of the norm regularizer (L1 or L2)
 *  @param lambda the amplitude of the regularization term
 *  @param conv_W a standard vector of weight matrices (one per conv layer)
 *  @param conv_B a standard vector of bias vectors (one per conv layer)
 */
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

/**
 *  \brief Implementation of the update function of the convolution layers 
 *
 *  @param batch_size number of examples in the batch
 *  @param eta the gradient descent step-size
 *  @param conv_gradB a standard vector of the gradient w.r.t. the pre-activation values (one per conv layer)
 *  @param conv_A a standard vector of activation values (one per conv layer)
 *  @param X_batch a matrix containing the training examples of the current minibatch
 *  @param regularizer the string of the norm regularizer (L1 or L2)
 *  @param lambda the amplitude of the regularization term
 *  @param conv_W a standard vector of weight matrices (one per conv layer)
 *  @param conv_B a standard vector of bias vectors (one per conv layer)
 */
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

/**
 *  \brief Implementation of the update function of the convolution layers 
 *
 *  @param batch_size number of examples in the batch
 *  @param eta the gradient descent step-size
 *  @param conv_gradB a standard vector of the gradient w.r.t. the pre-activation values (one per conv layer)
 *  @param conv_A a standard vector of activation values (one per conv layer)
 *  @param X_batch a matrix containing the training examples of the current minibatch
 *  @param regularizer the string of the norm regularizer (L1 or L2)
 *  @param lambda the amplitude of the regularization term
 *  @param conv_W a standard vector of weight matrices (one per conv layer)
 *  @param conv_B a standard vector of bias vectors (one per conv layer)
 */
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

void computeLoss(const ActivationFunction &act_func,
				 const unsigned batch_size,
				 const MyMatrix &X,
				 const MyVector &Y,
				 const std::vector<ConvLayerParams> &conv_params,
				 const std::vector<PoolLayerParams> &pool_params,
				 const std::vector<MyMatrix> &conv_W,
				 const std::vector<MyVector> &conv_B,
				 const std::vector<MyMatrix> &W, 
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
  
	a = (a*W[W.size()-1]).rowwise() + B[W.size()-1].transpose();

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
			   const unsigned batch_size,
			   const unsigned n_example,
			   const MyMatrix &X,
			   const MyVector &Y,
			   const std::vector<ConvLayerParams> &conv_params,
			   const std::vector<PoolLayerParams> &pool_params,
			   const std::vector<MyMatrix> &conv_W,
			   const std::vector<MyVector> &conv_B,
			   std::vector<MyMatrix> &W_eval,
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
		getMiniBatch(j, batch_size, X, Y, params, conv_params[0], curr_batch_size, X_batch, Y_batch);
    
		double loss = 0.;
		double accuracy = 0.;
		computeLoss(eval_act_func, curr_batch_size, X_batch, Y_batch, conv_params, pool_params, conv_W, conv_B, W_eval, B, loss, accuracy);

		acc_loss += loss;
		acc_accuracy += accuracy;
	}
  
	acc_loss/=n_example;
	acc_accuracy/=n_example;
}

void evalModel(const ActivationFunction &eval_act_func,
			   const unsigned train_batch_size,
			   const MyMatrix &X_train,
			   const MyMatrix &Y_train,
			   const unsigned valid_batch_size,
			   const MyMatrix &X_valid,
			   const MyMatrix &Y_valid,
			   const std::vector<ConvLayerParams> &conv_params,
			   const std::vector<PoolLayerParams> &pool_params,
			   const std::vector<MyMatrix> &conv_W,
			   const std::vector<MyVector> &conv_B,
			   std::vector<MyMatrix> &W_eval,
			   std::vector<MyVector> &B,
			   double &train_loss,
			   double &train_accuracy,
			   double &valid_loss,
			   double &valid_accuracy){
    
	// Training accuracy 
	computeLoss(eval_act_func, train_batch_size, X_train, Y_train, conv_params, pool_params, conv_W, conv_B, W_eval, B, train_loss, train_accuracy);
  
	// Validation accuracy
	computeLoss(eval_act_func, valid_batch_size, X_valid, Y_valid, conv_params, pool_params, conv_W, conv_B, W_eval, B, valid_loss, valid_accuracy);

}
