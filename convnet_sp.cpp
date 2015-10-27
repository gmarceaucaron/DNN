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
 * \file convnet_sp.cpp
 * \brief main function for launching experiments with convolutional neural networks and sparse network
 * \author Gaetan Marceau Caron & Yann Ollivier
 * \version 1.0
 */

#include "utils.hpp"
#include "nn_ops_sp.hpp"

int main(int argc, char *argv[]){
  
	Params params;
  
	std::map<std::string, std::string> args;
	readArgs(argc, argv, args);
	if(args.find("algo")!=args.end()){
		params.algo = args["algo"];
	}else{
		params.algo = "qdMCNat";
	}

	if(args.find("inst_file")!=args.end())
		setParamsFromFile(args["inst_file"], args, params);
	else   
		setParams(params.algo, args, params);
  
	createLogDir(params.dir_path);
  
	gen.seed(params.seed);

	// Load the dataset
	MyMatrix X_train, X_valid;
	VectorXd Y_train, Y_valid;
	loadMnist(params.ratio_train, X_train, X_valid, Y_train, Y_valid);
	//loadCIFAR10(params.ratio_train, X_train, X_valid, Y_train, Y_valid);
	//loadLightCIFAR10(params.ratio_train, X_train, X_valid, Y_train, Y_valid);
  
	// ConvNet parameters
	std::vector<ConvLayerParams> conv_params;
	ConvLayerParams conv_params1;
	conv_params1.Hf = 5;
	conv_params1.stride = 1;
	conv_params1.n_filter = 20;
	conv_params1.padding = 0;
	conv_params.push_back(conv_params1);
  
	ConvLayerParams conv_params2;
	conv_params2.Hf = 5;
	conv_params2.stride = 1;
	conv_params2.n_filter = 50;
	conv_params2.padding = 0;
	conv_params.push_back(conv_params2);

	std::vector<PoolLayerParams> pool_params;
	PoolLayerParams pool_params1;
	pool_params1.Hf = 2;
	pool_params1.stride = 2;
	pool_params.push_back(pool_params1);

	PoolLayerParams pool_params2;
	pool_params2.Hf = 2;
	pool_params2.stride = 2;
	pool_params.push_back(pool_params2);
  
	const unsigned n_conv_layer = conv_params.size();
  
	for(unsigned l = 0; l < conv_params.size(); l++){

		if(l==0){
			conv_params[l].filter_size = conv_params[l].Hf * conv_params[l].Hf * params.img_depth;
			conv_params[l].N = (params.img_width - conv_params[l].Hf + 2*conv_params[l].padding)/conv_params[l].stride + 1;
		}
		else{
			conv_params[l].filter_size = conv_params[l].Hf * conv_params[l].Hf * conv_params[l-1].n_filter;
			conv_params[l].N = (pool_params[l-1].N - conv_params[l].Hf + 2*conv_params[l].padding)/conv_params[l].stride + 1;
		}
		pool_params[l].N = (conv_params[l].N - pool_params[l].Hf)/pool_params[l].stride + 1;
	}
  
	// Neural Network parameters
	const unsigned n_training = X_train.rows();
	const unsigned n_valid = X_valid.rows();
	const unsigned n_feature = X_train.cols();
	const unsigned n_label = Y_train.maxCoeff() + 1;
  
	params.nn_arch.insert(params.nn_arch.begin(),conv_params[n_conv_layer-1].n_filter * pool_params[n_conv_layer-1].N * pool_params[n_conv_layer-1].N);
	params.nn_arch.push_back(n_label);
	const unsigned n_layers = params.nn_arch.size();
  
	// Optimization parameter
	const int n_train_batch = ceil(n_training/(float)params.train_minibatch_size);
	const int n_valid_batch = ceil(n_valid/(float)params.valid_minibatch_size);
	double prev_loss = std::numeric_limits<double>::max();
	double eta = params.eta;

	// Create the convolutional layer
	std::vector<MyMatrix> conv_W(n_conv_layer);
	std::vector<MyMatrix> conv_W_T(n_conv_layer);
	std::vector<MyVector> conv_B(n_conv_layer);
  
	// Create the neural network
	MyMatrix W_out(params.nn_arch[n_layers-2],n_label);
	std::vector<MySpMatrix> W(n_layers-2);
	std::vector<MySpMatrix> Wt(n_layers-2);
	std::vector<MyVector> B(n_layers-1);

	double init_sigma = 0.;
	ActivationFunction act_func;
	ActivationFunction eval_act_func;
	if(params.act_func_name=="sigmoid"){
		init_sigma = 4.0;
		act_func = std::bind(logistic,true,_1,_2,_3);
		eval_act_func = std::bind(logistic,false,_1,_2,_3);
	}else if(params.act_func_name=="tanh"){
		init_sigma = 1.0;
		act_func = std::bind(my_tanh,true,_1,_2,_3);
		eval_act_func = std::bind(my_tanh,false,_1,_2,_3);
	}else if(params.act_func_name=="relu"){
		init_sigma = 1.0; // TODO: Find the good value
		act_func = std::bind(relu,true,_1,_2,_3);
		eval_act_func = std::bind(relu,false,_1,_2,_3);
	}else{
		std::cout << "Not implemented yet!" << std::endl;
		assert(false);
	}

	std::cout << "Initializing the network... ";
	params.n_params = initNetwork(params.nn_arch, params.act_func_name, params.sparsity, conv_params, pool_params, W_out, W, Wt, B, conv_W, conv_W_T, conv_B); // TODO: Init the conv bias

	// Deep copy of parameters for the adaptive rule
	std::vector<MyMatrix> mu_dW(n_layers-1);
	std::vector<MyVector> mu_dB(n_layers-1);

	MyMatrix pW_out = W_out;
	std::vector<MySpMatrix> pW = W;
	std::vector<MySpMatrix> pWt = Wt;
	std::vector<MyVector> pB = B;

	MyMatrix ppMii_out, ppM0i_out;
	MyVector ppM00_out;
  
	std::vector<MySpMatrix> ppMii,ppM0i;
	std::vector<MyVector> ppM00;

	MyMatrix pMii_out,pM0i_out;
	MyVector pM00_out;
  
	std::vector<MySpMatrix> pMii,pM0i;
	std::vector<MyVector> pM00;

	std::vector<MyMatrix> conv_ppMii, conv_ppM0i;
	std::vector<MyVector> conv_ppM00;

	std::vector<MyMatrix> conv_pMii, conv_pM0i;
	std::vector<MyVector> conv_pM00;
  
	// Convert the labels to one-hot vector
	MyMatrix one_hot = MyMatrix::Zero(n_training, n_label);
	labels2oneHot(Y_train,one_hot);
  
	// Configure the logger 
	std::ostream* logger;
	if(args.find("verbose")!=args.end()){
		getOutput("",logger);
	}else{
		getOutput(params.file_path,logger);
	}

	double cumul_time = 0.;
  
	printDesc(params, logger);
	printConvDesc(params, conv_params, pool_params, logger);
	std::cout << "Starting the learning phase... " << std::endl;
	*logger << "Epoch Time(s) train_loss train_accuracy valid_loss valid_accuracy eta" << std::endl;
  
	for(unsigned i = 0; i < params.n_epoch; i++){
		for(unsigned j = 0; j < n_train_batch; j++){
      
			// Mini-batch creation
			unsigned curr_batch_size = 0;
			MyMatrix X_batch, one_hot_batch;
			getMiniBatch(j, params.train_minibatch_size, X_train, one_hot, params, conv_params[0], curr_batch_size, X_batch, one_hot_batch);
      
			double prev_time = gettime();

			// Forward propagation for conv layer
			std::vector<std::vector<unsigned>> poolIdxX1(n_conv_layer);
			std::vector<std::vector<unsigned>> poolIdxY1(n_conv_layer);
      
			MyMatrix z0;
			std::vector<MyMatrix> conv_A(conv_W.size());
			std::vector<MyMatrix> conv_Ap(conv_W.size());
			convFprop(curr_batch_size, conv_params, pool_params, act_func, conv_W, conv_B, X_batch, conv_A, conv_Ap, z0, poolIdxX1, poolIdxY1);
            
			// Forward propagation
			std::vector<MyMatrix> Z(n_layers-1);
			std::vector<MyMatrix> A(n_layers-2);
			std::vector<MyMatrix> Ap(n_layers-2);
			fprop(params.dropout_flag, act_func, W, W_out, B, z0, Z, A, Ap);
      
			// Compute the output and the error
			MyMatrix out;
			softmax(Z[n_layers-2], out);
      
			std::vector<MyMatrix> gradB(n_layers-1);
			gradB[n_layers-2] = out - one_hot_batch;

			// Backpropagation
			bprop(Wt, W_out, Ap, gradB);

			// Backpropagation for conv layer
			std::vector<MyMatrix> conv_gradB(conv_W.size());
			MyMatrix layer_gradB = (gradB[0] * W[0].transpose());
			MyMatrix pool_gradB;
			layer2pool(curr_batch_size, pool_params[conv_W.size()-1].N, conv_params[conv_W.size()-1].n_filter, layer_gradB, pool_gradB);
      
			convBprop(curr_batch_size, conv_params, pool_params, conv_W_T, conv_Ap, pool_gradB, conv_gradB, poolIdxX1, poolIdxY1);
      
			if(params.algo == "bprop"){
				update(eta, gradB, A, z0, params.regularizer, params.lambda, W_out, W, Wt, B);
				convUpdate(curr_batch_size, eta, conv_params, conv_gradB, conv_A, X_batch, "", 0., conv_W, conv_W_T, conv_B);
	
			}else{

				// Compute the metric
				std::vector<MyMatrix> metric_gradB(n_layers-1);
				std::vector<MyMatrix> metric_conv_gradB(conv_params.size());

				if(params.algo=="qdMCNat"){

					// Monte-Carlo Approximation of the metric
					std::vector<MyMatrix> mc_gradB(n_layers-1);
					computeMcError(out, mc_gradB[n_layers-2]);

					// Backpropagation
					bprop(Wt, W_out, Ap, mc_gradB);

					for(unsigned k = 0; k < gradB.size(); k++){
						metric_gradB[k] = mc_gradB[k].array().square();
					}

					// Backpropagation for conv layer
					std::vector<MyMatrix> mc_conv_gradB(conv_W.size());
					MyMatrix mc_layer_gradB = (mc_gradB[0] * W[0].transpose());
					MyMatrix mc_pool_gradB;
					layer2pool(curr_batch_size, pool_params[conv_W.size()-1].N, conv_params[conv_W.size()-1].n_filter, mc_layer_gradB, mc_pool_gradB);
	  
					convBprop(curr_batch_size, conv_params, pool_params, conv_W_T, conv_Ap, mc_pool_gradB, mc_conv_gradB, poolIdxX1, poolIdxY1);
	  
					for(unsigned k = 0; k < conv_params.size(); k++){
						metric_conv_gradB[k] = mc_conv_gradB[k].array().square();
					}
				}	
				else if(params.algo=="qdop"){

					for(unsigned k = 0; k < conv_params.size(); k++){
						metric_conv_gradB[k] = conv_gradB[k].array().square();
					}
					for(unsigned k = 0; k < gradB.size(); k++){
						metric_gradB[k] = gradB[k].array().square();
					}
				}
				else if(params.algo=="qdNat"){
	  
					for(unsigned k = 0; k < conv_params.size(); k++){
						metric_conv_gradB[k] = conv_gradB[k].array().square();
					}

					for(unsigned k = 0; k < metric_gradB.size(); k++){
						metric_gradB[k] = MyMatrix::Zero(gradB[k].rows(),gradB[k].cols());
					}

					for(unsigned l = 0; l < n_label; l++){
						MyMatrix fisher_ohbatch = MyMatrix::Zero(curr_batch_size, n_label);
						fisher_ohbatch.col(l).setOnes();

						std::vector<MyMatrix> fgradB(n_layers-1);
						fgradB[n_layers-2] = out - fisher_ohbatch;
						bprop(Wt, W_out, Ap, fgradB);

						// Backpropagation for conv layer
						std::vector<MyMatrix> fisher_conv_gradB(conv_W.size());
						MyMatrix fisher_layer_gradB = (fgradB[0] * W[0].transpose());
						MyMatrix fisher_pool_gradB;
						layer2pool(curr_batch_size, pool_params[conv_W.size()-1].N, conv_params[conv_W.size()-1].n_filter, fisher_layer_gradB, fisher_pool_gradB);
	    
						convBprop(curr_batch_size, conv_params, pool_params, conv_W_T, conv_Ap, fisher_pool_gradB, fisher_conv_gradB, poolIdxX1, poolIdxY1);

						for(unsigned k = 0; k < conv_params.size(); k++){
							MyMatrix fisher_conv_gradB_sq = fisher_conv_gradB[k].array().square();
							for(unsigned m = 0; m < out.rows(); m++){
								for(unsigned f = 0; f < conv_params[k].n_filter; f++){
									for(unsigned n = 0; n < conv_params[k].N * conv_params[k].N; n++){
										fisher_conv_gradB_sq(f,m*conv_params[k].N*conv_params[k].N+n) *= out(m,l);
									}
								}
							}
							metric_conv_gradB[k] += fisher_conv_gradB_sq;
						}
	    
						for(unsigned k = 0; k < W.size(); k++){
							const unsigned rev_k = n_layers - k - 2;
							metric_gradB[rev_k] += (fgradB[rev_k].array().square().array().colwise() * out.array().col(l)).matrix();
						}
					}
				}
	
				bool init_flag = false;
				if(i == 0 && j == 0 && !params.init_metric_id){
					init_flag = true;
				}

				std::vector<MyMatrix> conv_Mii(conv_params.size());
				std::vector<MyMatrix> conv_M0i(conv_params.size());
				std::vector<MyVector> conv_M00(conv_params.size());
	
				buildConvQDMetric(curr_batch_size, metric_conv_gradB, conv_A, X_batch, conv_W, params.matrix_reg, conv_Mii, conv_M0i, conv_M00);

				updateConvMetric(init_flag, params.metric_gamma, conv_pMii, conv_pM0i, conv_pM00, conv_Mii, conv_M0i, conv_M00);

				MyMatrix Mii_out, M0i_out;
				MyVector M00_out;
				std::vector<MySpMatrix> Mii(W.size());
				std::vector<MySpMatrix> M0i(W.size());
				std::vector<MyVector> M00(W.size());

				buildQDMetric(metric_gradB, A, z0, W_out, W, params.matrix_reg, Mii_out, M0i_out, M00_out, Mii, M0i, M00);

				updateMetric(init_flag, params.metric_gamma, Mii_out, M0i_out, M00_out, Mii, M0i, M00, pMii_out, pM0i_out, pM00_out, pMii, pM0i, pM00);
				update(eta, gradB, A, z0, params.regularizer, params.lambda, W_out, W, Wt, B, Mii_out, M0i_out, M00_out, Mii, M0i, M00);
			}
      
			double curr_time = gettime();
			cumul_time += curr_time - prev_time;      
      
			if(params.minilog_flag){
	
				double train_loss = 0.;
				double train_accuracy = 0.;
				double valid_loss = 0.;
				double valid_accuracy = 0.;
				evalModel(eval_act_func, params, n_train_batch, n_training, X_train, Y_train, conv_params, pool_params, conv_W, conv_B, W_out, W, B, train_loss, train_accuracy);
				evalModel(eval_act_func, params, n_valid_batch, n_valid, X_valid, Y_valid, conv_params, pool_params, conv_W, conv_B, W_out, W, B, valid_loss, valid_accuracy);
	
				// Logging
				*logger << i + float(j)/n_train_batch << " " << cumul_time << " " << train_loss <<  " " << train_accuracy << " " << valid_loss <<  " " << valid_accuracy << " " << eta << std::endl;
	
			}
		}
		if(!params.minilog_flag || params.adaptive_flag){
			double train_loss = 0.;
			double train_accuracy = 0.;
			double valid_loss = 0.;
			double valid_accuracy = 0.;
			evalModel(eval_act_func, params, n_train_batch, n_training, X_train, Y_train, conv_params, pool_params, conv_W, conv_B, W_out, W, B, train_loss, train_accuracy);
			evalModel(eval_act_func, params, n_valid_batch, n_valid, X_valid, Y_valid, conv_params, pool_params, conv_W, conv_B, W_out, W, B, valid_loss, valid_accuracy);
      
			// if(params.adaptive_flag)
			// 	adaptiveRule(train_loss, prev_loss, eta, W, B, pMii, pM0i, pM00, pW, pB, ppMii, ppM0i, ppM00);
      
			// Logging
			if(!params.minilog_flag){
				*logger << i  << " " << cumul_time << " " << train_loss <<  " " << train_accuracy << " " << valid_loss <<  " " << valid_accuracy << " " << eta << std::endl;
			}
		}
	}
}
