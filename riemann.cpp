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
#include "utils.hpp"
#include "nn_ops.hpp"

/*!
 * \file riemann.cpp
 * \brief main function for launching experiments with neural networks
 * \author Gaetan Marceau Caron & Yann Ollivier
 * \version 1.0
 */
int main(int argc, char *argv[]){

	// Parameter handling
	Params params;  
	std::map<std::string, std::string> args;
	readArgs(argc, argv, args);
	if(args.find("algo")!=args.end()){
		params.algo = args["algo"];
	}else{
		params.algo = "bprop";
	}

	if(args.find("inst_file")!=args.end())
		setParamsFromFile(args["inst_file"], args, params);
	else   
		setParams(params.algo, args, params);

	// Create the directory for log files
	createLogDir(params.dir_path);

	// Set the seed
	gen.seed(params.seed);
  
	// Load the dataset 
	MyMatrix X_train, X_valid;
	VectorXd Y_train, Y_valid;
	loadMnist(params, X_train, X_valid, Y_train, Y_valid);
	const unsigned n_training = X_train.rows();
	const unsigned n_feature = X_train.cols();
	const unsigned n_label = Y_train.maxCoeff() + 1;
	
	// Neural Network parameters
	params.nn_arch.insert(params.nn_arch.begin(),n_feature);
	params.nn_arch.push_back(n_label);
	const unsigned n_layers = params.nn_arch.size();
  
	// Optimization parameter
	const int n_batch = ceil(n_training/(float)params.minibatch_size);
	double prev_loss = std::numeric_limits<double>::max();
	double eta = params.eta;

	// Create the data structure for the parameters
	std::vector<MyMatrix> W(n_layers-1);
	std::vector<MyVector> B(n_layers-1);

	// Declare the activation function and fix the initialization parameter
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
	params.n_params = initNetwork(params.nn_arch, params.act_func_name, W, B);

	// Deep copy of the parameters for the adaptive rule
	std::vector<MyMatrix> mu_dW(n_layers-1);
	std::vector<MyVector> mu_dB(n_layers-1);

	std::vector<MyMatrix> pW = W;
	std::vector<MyVector> pB = B;

	std::vector<MyMatrix> ppMii,ppM0i;
	std::vector<MyVector> ppM00;

	std::vector<MyMatrix> pMii,pM0i;
	std::vector<MyVector> pM00;

	if(params.init_metric_id)
		initSlidingMetric(W, pMii, pM0i, pM00);

	// Convert the labels to one-hot representation
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
	std::cout << "Starting the learning phase... " << std::endl;
	*logger << "Epoch Time(s) train_loss train_accuracy valid_loss valid_accuracy eta" << std::endl;
  
	for(unsigned i = 0; i < params.n_epoch; i++){
		for(unsigned j = 0; j < n_batch; j++){
      
			// Mini-batch creation
			unsigned curr_batch_size = 0;
			MyMatrix X_batch, one_hot_batch;
			getMiniBatch(j, params.minibatch_size, X_train, Y_train, one_hot, curr_batch_size, X_batch, one_hot_batch);

			double prev_time = gettime();
      
			// Forward propagation
			std::vector<MyMatrix> Z(n_layers-1);
			std::vector<MyMatrix> A(n_layers-2);
			std::vector<MyMatrix> Ap(n_layers-2);
			fprop(params.dropout_flag, params.dropout_prob, act_func, W, B, X_batch, Z, A, Ap);
      
			// Compute the output and the error
			MyMatrix out;
			softmax(Z[n_layers-2], out);

			std::vector<MyMatrix> gradB(n_layers-1);
			gradB[n_layers-2] = out - one_hot_batch;
      
			// Backpropagation
			bprop(W, Ap, gradB);

			// The following depends on the chosen algorithm
			if(params.algo=="bprop"){

				// Simply update the model parameters
				update(eta, gradB, A, X_batch, params.regularizer, params.lambda, W, B);

			}else if(params.algo=="adagrad"){

				// Update the model parameters with the adagrad rule
				adagradUpdate(eta, gradB, A, X_batch, params.regularizer, params.lambda, params.matrix_reg, 1.0-params.metric_gamma, W, B, mu_dW, mu_dB);
				
			}else{
				// The following algorithms compute a riemannian metric

				// Data structure necessary for computing the metric
				std::vector<MyMatrix> metric_gradB(n_layers-1);
		  
				if(params.algo=="qdMCNat"){

					// Monte-Carlo Approximation of the metric
					std::vector<MyMatrix> mc_gradB(n_layers-1);

					// Two possible Monte-Carlo schemes
					if(params.imp_sampling_ratio>0.){
						computeISError(out, one_hot_batch, params.imp_sampling_ratio, mc_gradB[n_layers-2]);
					}else{
						computeMcError(out, mc_gradB[n_layers-2]);
					}
	  
					// Backpropagation
					bprop(W, Ap, mc_gradB);

					// Update
					for(unsigned k = 0; k < gradB.size(); k++){
						metric_gradB[k] = mc_gradB[k].array().square();
					}
				}
				else if(params.algo=="qdLazyNat"){
					std::vector<MyMatrix> lazy_gradB(n_layers-1);

					// Lazy metric (avoid this!)
					computeLazyError(out, one_hot_batch, lazy_gradB[n_layers-2]);
	  
					// Backpropagation
					bprop(W, Ap, lazy_gradB);

					// Update
					for(unsigned k = 0; k < gradB.size(); k++){
						metric_gradB[k] = lazy_gradB[k].array().square();
					}
				}
				else if(params.algo=="qdop" || params.algo=="dop"){

					// Update
					for(unsigned k = 0; k < gradB.size(); k++){
						metric_gradB[k] = gradB[k].array().square();
					}
					
				}else if(params.algo=="qdNat"){

					// Update for each label weighted by the output probability
					for(unsigned k = 0; k < metric_gradB.size(); k++){
						metric_gradB[k] = MyMatrix::Zero(gradB[k].rows(),gradB[k].cols());
					}
	  
					for(unsigned l = 0; l < n_label; l++){
						MyMatrix fisher_ohbatch = MyMatrix::Zero(curr_batch_size, n_label);
						fisher_ohbatch.col(l).setOnes();
	    
						std::vector<MyMatrix> fgradB(n_layers-1);
						fgradB[n_layers-2] = out - fisher_ohbatch;
						bprop(W, Ap, fgradB);
	    
						for(unsigned i = 0; i < W.size(); i++){
							const unsigned rev_i = n_layers - i - 2;
							metric_gradB[rev_i] += (fgradB[rev_i].array().square().array().colwise() * out.array().col(l)).matrix();
						}
					}
				}else if(params.algo=="qdbpm" || params.algo=="gaussNewton"){
					metric_gradB[n_layers-2] = (out.array() * (1.0-out.array())).matrix();
					qdBpmBprop(W, Ap, metric_gradB);
				}else{
					std::cout << "Not implemented yet!" << std::endl;
					assert(false);
				}

				// Except for these two algorithms, the others require the following
				if(params.algo!="gaussNewton" && params.algo!="dop"){

					// Data structure for quasi-diagonal metric
					std::vector<MyMatrix> Mii(W.size());
					std::vector<MyMatrix> M0i(W.size());
					std::vector<MyVector> M00(W.size());

					// Build the quasi-diagonal metric
					buildQDMetric(metric_gradB, A, X_batch, W, params.matrix_reg, Mii, M0i, M00);

					// Update the metric as a sliding average
					bool init_flag = false;
					if(i == 0 && j == 0 && !params.init_metric_id){
						init_flag = true;
					}
					updateMetric(init_flag, params.metric_gamma, Mii, M0i, M00, pMii, pM0i, pM00);	

					// Update the parameters with the natural gradient descent
					update(eta, gradB, A, X_batch, params.regularizer, params.lambda, W, B, Mii, M0i, M00);

				}else{

					// These are diagonal metric approximation

					// Data structure for diagonal metric
					std::vector<MyMatrix> Mii(W.size());
					std::vector<MyVector> M00(W.size());

					// Build the diagonal metric
					buildDiagMetric(metric_gradB, A, X_batch, W, params.matrix_reg, Mii, M00);

					// Update the metric as a sliding average
					bool init_flag = false;
					if(i == 0 && j == 0 && !params.init_metric_id){
						init_flag = true;
					}
					updateMetric(init_flag, params.metric_gamma, Mii, M00, pMii, pM00);	

					// Update the parameters with the natural gradient descent
					update(eta, gradB, A, X_batch, params.regularizer, params.lambda, W, B, Mii, M00);
				}
			}

			double curr_time = gettime();
			cumul_time += curr_time - prev_time;      

			// If minilog is activated, compute the NLL on the whole dataset
			if(params.minilog_flag){
	
				double train_loss = 0.;
				double train_accuracy = 0.;
				double valid_loss = 0.;
				double valid_accuracy = 0.;
				evalModel(eval_act_func, X_train, Y_train, X_valid, Y_valid, params, W, B, train_loss, train_accuracy, valid_loss, valid_accuracy);
	
				// Logging
				*logger << i + float(j)/n_batch << " " << cumul_time << " " << train_loss <<  " " << train_accuracy << " " << valid_loss <<  " " << valid_accuracy << " " << eta << std::endl;
	
			}
		}

		// Compute the NLL on the whole dataset
		if(!params.minilog_flag || params.adaptive_flag){
			double train_loss = 0.;
			double train_accuracy = 0.;
			double valid_loss = 0.;
			double valid_accuracy = 0.;
			evalModel(eval_act_func, X_train, Y_train, X_valid, Y_valid, params, W, B, train_loss, train_accuracy, valid_loss, valid_accuracy);

			// Perform step-size adaptation
			if(params.adaptive_flag)
				adaptiveRule(train_loss, prev_loss, eta, W, B, pMii, pM0i, pM00, pW, pB, ppMii, ppM0i, ppM00);
			else if(params.schedule_flag){
				updateStepsize(params.minibatch_size * n_batch * (i + 1), params.eta, eta);
			}
      
			// Logging
			if(!params.minilog_flag){
				*logger << i  << " " << cumul_time << " " << train_loss <<  " " << train_accuracy << " " << valid_loss <<  " " << valid_accuracy << " " << eta << " " << std::endl;
			}
		}
	}
	return EXIT_SUCCESS;
}
