#include "utils.hpp"
#include "nn_ops.hpp"

int main(int argc, char *argv[]){
	
	// Finite Difference parameters
	const double EPS = 1e-6;
	const unsigned n_datapoint = 10;
	const unsigned burn_in_epoch = 2;

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
	
	// Set the seed for the random generator
	unsigned seed = 0;
	gen.seed(seed);
	
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

	// Burn-in period
	for(unsigned i = 0; i < burn_in_epoch; i++){
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

			// Simply update the model parameters
			update(eta, gradB, A, X_batch, params.regularizer, params.lambda, W, B);
      
		}
	}

	for(unsigned ex_id = 0; ex_id < n_datapoint; ex_id++){
		for(unsigned l = 1; l < W.size(); l++){
		  
			// Test the weights
			for(unsigned  i = 0; i < W[l].rows(); i++){
				for(unsigned  j = 0; j < W[l].cols(); j++){
				  
					// Change a parameter
					W[l](i,j) += EPS;
				  
					// Forward propagation
					std::vector<MyMatrix> Z1(n_layers-1);
					std::vector<MyMatrix> A1(n_layers-2);
					std::vector<MyMatrix> Ap1(n_layers-2);
					//fprop(false, act_func, W, B, X_train.row(ex_id), Z1, A1, Ap1);
					fprop(false, 0., act_func, W, B, X_train.row(ex_id), Z1, A1, Ap1);
					
					// Compute the output and the error
					MyMatrix out1;
					softmax(Z1[n_layers-2], out1);
					MyMatrix error1 = out1 - one_hot.row(ex_id);
				  
					// Change a parameter
					W[l](i,j) -= 2.0 * EPS;
				  
					// Forward propagation
					std::vector<MyMatrix> Z2(n_layers-1);
					std::vector<MyMatrix> A2(n_layers-2);
					std::vector<MyMatrix> Ap2(n_layers-2);
					//fprop(false, act_func, W, B, X_train.row(ex_id), Z2, A2, Ap2);
					fprop(false, 0., act_func, W, B, X_train.row(ex_id), Z2, A2, Ap2);
				  
					// Compute the output and the error
					MyMatrix out2;
					softmax(Z2[n_layers-2], out2);
					MyMatrix error2 = out2 - one_hot.row(ex_id);
				  
					// Compute the update with finite differences 
					for(unsigned i = 0; i < error1.cols(); i++){
						out1(i) = log(out1(i));
						out2(i) = log(out2(i));
					}
					MyMatrix all_delta_errors = (out2-out1)/(2.0 * EPS);
	  
					// Change a parameter
					W[l](i,j) += EPS;
	  
					// Forward propagation 
					std::vector<MyMatrix> Z3(n_layers-1);
					std::vector<MyMatrix> A3(n_layers-2);
					std::vector<MyMatrix> Ap3(n_layers-2);
					//fprop(false, act_func, W, B, X_train.row(ex_id), Z3, A3, Ap3);
					fprop(false, 0., act_func, W, B, X_train.row(ex_id), Z3, A3, Ap3);
					
					// Compute the output and the error 
					MyMatrix out3;
					std::vector<MyMatrix> gradB(n_layers-1);
					softmax(Z3[n_layers-2], out3);
					gradB[n_layers-2] = out3 - one_hot.row(ex_id);
	  
					// Backpropagation
					bprop(W, Ap3, gradB);
	  
					// Update 
					std::vector<MyMatrix> DW;
					std::vector<MyVector> DB;
					testUpdate(eta, gradB, A3, X_train.row(ex_id), "", 0., W, B, DW, DB);
	  
					double approx_dw = 0.0;
					for(unsigned i = 0; i < all_delta_errors.rows(); i++){
						approx_dw += all_delta_errors(i,Y_train(ex_id));
					}

					if(abs(approx_dw) > 1e-6){
						const double rel_error = abs(DW[l](i,j) - approx_dw) / (abs(approx_dw)+abs(DW[l](i,j)));
						std::cout << "Weight Real-Approx-RelError " << DW[l](i,j) << " " << approx_dw << " (" << rel_error << ")" << std::endl;
						if(rel_error > 0.01){
							std::cout << "weight testing failed " << ex_id << " " << l << " " << i << " " << j << " " << rel_error << " " << std::endl;
							assert(false);
						}
					}
				}
			}

			// Test the bias
			for(unsigned  i = 0; i < B.size(); i++){

				// Change a parameter
				B[l](i) += EPS;
	  
				// Forward propagation
				std::vector<MyMatrix> Z1(n_layers-1);
				std::vector<MyMatrix> A1(n_layers-2);
				std::vector<MyMatrix> Ap1(n_layers-2);
				fprop(false, 0., act_func, W, B, X_train.row(ex_id), Z1, A1, Ap1);
	
				// Compute the output and the error
				MyMatrix out1;
				softmax(Z1[n_layers-2], out1);
				MyMatrix error1 = out1 - one_hot.row(ex_id);
	
				// Change a parameter
				B[l](i) -= 2.0 * EPS;
	  
				// Forward propagation
				std::vector<MyMatrix> Z2(n_layers-1);
				std::vector<MyMatrix> A2(n_layers-2);
				std::vector<MyMatrix> Ap2(n_layers-2);
				fprop(false, 0., act_func, W, B, X_train.row(ex_id), Z2, A2, Ap2);
	
				// Compute the output and the error
				MyMatrix out2;
				softmax(Z2[n_layers-2], out2);
				MyMatrix error2 = out2 - one_hot.row(ex_id);
	
				// Compute the update with finite differences 
				for(unsigned i = 0; i < error1.cols(); i++){
					out1(i) = log(out1(i));
					out2(i) = log(out2(i));
				}
				MyMatrix all_delta_errors = (out2-out1)/(2.0 * EPS);
	
				// Change a parameter
				B[l](i) += EPS;
	
				// Forward propagation 
				std::vector<MyMatrix> Z3(n_layers-1);
				std::vector<MyMatrix> A3(n_layers-2);
				std::vector<MyMatrix> Ap3(n_layers-2);
				fprop(false, 0., act_func, W, B, X_train.row(ex_id), Z3, A3, Ap3);
	
				// Compute the output and the error 
				MyMatrix out3;
				std::vector<MyMatrix> gradB(n_layers-1);
				softmax(Z3[n_layers-2], out3);
				gradB[n_layers-2] = out3 - one_hot.row(ex_id);
	
				// Backpropagation
				bprop(W, Ap3, gradB);
	
				// Update 
				std::vector<MyMatrix> DW;
				std::vector<MyVector> DB;
				testUpdate(eta, gradB, A3, X_train.row(ex_id), "", 0., W, B, DW, DB);
	
				double approx_db = 0.0;
				for(unsigned i = 0; i < all_delta_errors.rows(); i++){
					approx_db += all_delta_errors(i,Y_train(ex_id));
				}

				if(abs(approx_db) > 1e-6){
					const double rel_error = abs(DB[l](i) - approx_db) / (abs(approx_db)+abs(DB[l](i)));
					std::cout << "Bias: Real-Approx-RelError " << DB[l](i) << " " << approx_db << " (" << rel_error << ")" << std::endl;
					if(rel_error > 0.01){
						std::cout << "bias testing failed " << ex_id << " " << l << " " << i << " " << rel_error << " " << std::endl;
						assert(false);
					}
				}
			}
		}      
	}
}
