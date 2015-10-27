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
 * \file conv_fdTest.cpp
 * \brief main function for testing backpropagation for convolutional neural networks
 * \author Gaetan Marceau Caron & Yann Ollivier
 * \version 1.0
 */

#include "utils.hpp"
#include "utils_conv.hpp"
#include "nn_ops_conv.hpp"


int main(int argc, char *argv[]){
  
  // Finite Difference parameters
  const double EPS = 1e-6;
  const double REL_ERROR_THRESHOLD = 1e-2;
  const unsigned n_datapoint = 10;
  const unsigned burn_in_epoch = 0;
  
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
  
  //gen.seed(params.seed);
  gen.seed(0);
  
  // Load the dataset
  MyMatrix X_train, X_valid;
  VectorXd Y_train, Y_valid;
  loadLightCIFAR10(0.9, X_train, X_valid, Y_train, Y_valid);
  
  // ConvNet parameters
  std::vector<ConvLayerParams> conv_params;
  ConvLayerParams conv_params1;
  conv_params1.Hf = 3;
  conv_params1.stride = 1;
  conv_params1.n_filter = 16;
  conv_params1.padding = 0;
  conv_params.push_back(conv_params1);
  
  ConvLayerParams conv_params2;
  conv_params2.Hf = 3;
  conv_params2.stride = 1;
  conv_params2.n_filter = 8;
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

  std::cout << "image information" << std::endl;
  std::cout << "height " << params.img_height << std::endl;
  std::cout << "width " << params.img_width << std::endl;
  std::cout << "depth " << params.img_depth << std::endl;

  std::cout << "filter information" << std::endl;
  for(unsigned l = 0; l < conv_params.size(); l++){
    std::cout << "Conv Hf " << conv_params[l].Hf << std::endl;
    std::cout << "Conv stride " << conv_params[l].stride << std::endl;
    std::cout << "Conv padding " << conv_params[l].padding << std::endl;
    
    std::cout << "Pool hf " << pool_params[l].Hf << std::endl;
    std::cout << "Pool stride " << pool_params[l].stride << std::endl;
    
    std::cout << "n filter " << conv_params[l].n_filter << std::endl;
    std::cout << "filter size " << conv_params[l].filter_size << std::endl;
    std::cout << "conv N " << conv_params[l].N << std::endl;
    std::cout << "pool N " << pool_params[l].N << std::endl << std::endl;
  }
    
  // Neural Network parameters
  const unsigned n_training = X_train.rows();
  const unsigned n_valid = X_valid.rows();
  const unsigned n_feature = X_train.cols();
  const unsigned n_label = Y_train.maxCoeff() + 1;

  params.nn_arch.insert(params.nn_arch.begin(),conv_params[n_conv_layer-1].n_filter * pool_params[n_conv_layer-1].N * pool_params[n_conv_layer-1].N);
  //params.nn_arch.insert(params.nn_arch.begin(),n_feature);
  params.nn_arch.push_back(n_label);
  const unsigned n_layers = params.nn_arch.size();

  // Optimization parameter
  const int n_batch = ceil(n_training/(float)params.train_minibatch_size);
  double prev_loss = std::numeric_limits<double>::max();
  double eta = params.eta;

  // Create the convolutional layer
  std::vector<MyMatrix> conv_W(n_conv_layer);
  std::vector<MyMatrix> conv_W_T(n_conv_layer);
  std::vector<MyVector> conv_B(n_conv_layer);
  
  // Create the neural network
  std::vector<MyMatrix> W(n_layers-1);
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
  params.n_params = initNetwork(params.nn_arch, params.act_func_name, conv_params, pool_params, W, B, conv_W, conv_W_T, conv_B); // TODO: Init the conv bias

  // Convert the whole dataset into convolutional input
  MyMatrix conv_X_train;
  data2conv(X_train, n_training, params.img_width, params.img_height, params.img_depth, conv_params[0].N, conv_params[0].Hf, conv_params[0].padding, conv_params[0].stride, conv_X_train);

  MyMatrix conv_X_valid;
  data2conv(X_valid, n_valid, params.img_width, params.img_height, params.img_depth, conv_params[0].N, conv_params[0].Hf, conv_params[0].padding, conv_params[0].stride, conv_X_valid);
  
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
  std::cout << "Starting the learning phase... " << std::endl;
  *logger << "Epoch Time(s) train_loss train_accuracy valid_loss valid_accuracy eta" << std::endl;
  
  // Burn-in period
  for(unsigned i = 0; i < burn_in_epoch; i++){
    std::cout << "Burning step " << i << std::endl;
    
    for(unsigned j = 0; j < n_batch; j++){
      
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
      fprop(false, 0., act_func, W, B, z0, Z, A, Ap);
      
      // Compute the output and the error
      MyMatrix out;
      softmax(Z[n_layers-2], out);
      
      std::vector<MyMatrix> gradB(n_layers-1);
      gradB[n_layers-2] = out - one_hot_batch;
      
      // Backpropagation
      bprop(W, Ap, gradB);

      // Backpropagation for conv layer
      std::vector<MyMatrix> conv_gradB(conv_W.size());
      MyMatrix layer_gradB = (gradB[0] * W[0].transpose());
      MyMatrix pool_gradB;
      layer2pool(curr_batch_size, pool_params[conv_W.size()-1].N, conv_params[conv_W.size()-1].n_filter, layer_gradB, pool_gradB);

      convBprop(curr_batch_size, conv_params, pool_params, conv_W_T, conv_Ap, pool_gradB, conv_gradB, poolIdxX1, poolIdxY1);
      
      convUpdate(eta, conv_gradB, conv_A, X_batch, "", 0., conv_W, conv_B);
      
      double train_loss = 0.;
      double train_accuracy = 0.;
      double valid_loss = 0.;
      double valid_accuracy = 0.;
      evalModel(eval_act_func, n_training, conv_X_train, Y_train, n_valid, conv_X_valid, Y_valid, conv_params, pool_params, conv_W, conv_B, W, B, train_loss, train_accuracy, valid_loss, valid_accuracy);
      
      // Logging
      *logger << i + float(j)/n_batch << " " << cumul_time << " " << train_loss <<  " " << train_accuracy << " " << valid_loss <<  " " << valid_accuracy << " " << eta << std::endl;
      
    }
  }
  std::cout << "Burning phase done" << std::endl;
  
  for(unsigned ex_id = 0; ex_id < n_datapoint; ex_id++){
    
    std::cout << "Testing the convolutional weight" << std::endl;
    for(unsigned l = 0; l < conv_W.size(); l++){
      std::cout << "Testing the " << l << "th layer" << std::endl;
      for(unsigned  i = 0; i < conv_W[l].rows(); i++){
	for(unsigned  j = 0; j < conv_W[l].cols(); j++){
	  
	  // Change a parameter
	  conv_W[l](i,j) += EPS;
	  
	  // Forward propagation for conv layer
	  std::vector<std::vector<unsigned>> poolIdxX1(n_conv_layer);
	  std::vector<std::vector<unsigned>> poolIdxY1(n_conv_layer);
	  
	  MyMatrix z0_1;
	  std::vector<MyMatrix> conv_A1(conv_W.size());
	  std::vector<MyMatrix> conv_Ap1(conv_W.size());
	  convFprop(1, conv_params, pool_params, act_func, conv_W, conv_B, conv_X_train.block(0, 0, conv_params[0].filter_size, conv_params[0].N*conv_params[0].N), conv_A1, conv_Ap1, z0_1, poolIdxX1, poolIdxY1);

	  // Forward propagation
	  std::vector<MyMatrix> Z1(n_layers-1);
	  std::vector<MyMatrix> A1(n_layers-2);
	  std::vector<MyMatrix> Ap1(n_layers-2);
	  //fprop(false, act_func, W, B, X_train.row(ex_id), Z1, A1, Ap1);
	  fprop(false, 0., act_func, W, B, z0_1, Z1, A1, Ap1);
	  
	  // Compute the output and the error
	  MyMatrix out1;
	  softmax(Z1[n_layers-2], out1);
	  MyMatrix error1 = out1 - one_hot.row(ex_id);
	  
	  // Change a parameter
	  conv_W[l](i,j) -= 2.0 * EPS;
	  
	  // Forward propagation for conv layer
	  std::vector<std::vector<unsigned>> poolIdxX2(n_conv_layer);
	  std::vector<std::vector<unsigned>> poolIdxY2(n_conv_layer);

	  MyMatrix z0_2;
	  std::vector<MyMatrix> conv_A2(conv_W.size());
	  std::vector<MyMatrix> conv_Ap2(conv_W.size());
	  convFprop(1, conv_params, pool_params, act_func, conv_W, conv_B, conv_X_train.block(0, 0, conv_params[0].filter_size, conv_params[0].N*conv_params[0].N), conv_A2, conv_Ap2, z0_2, poolIdxX2, poolIdxY2);
	  	  
	  // Forward propagation
	  std::vector<MyMatrix> Z2(n_layers-1);
	  std::vector<MyMatrix> A2(n_layers-2);
	  std::vector<MyMatrix> Ap2(n_layers-2);
	  fprop(false, 0., act_func, W, B, z0_2, Z2, A2, Ap2);
	  
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
	  conv_W[l](i,j) += EPS;

	  // Forward propagation for conv layer
	  std::vector<std::vector<unsigned>> poolIdxX3(n_conv_layer);
	  std::vector<std::vector<unsigned>> poolIdxY3(n_conv_layer);

	  MyMatrix z0_3;
	  std::vector<MyMatrix> conv_A3(conv_W.size());
	  std::vector<MyMatrix> conv_Ap3(conv_W.size());
	  convFprop(1, conv_params, pool_params, act_func, conv_W, conv_B, conv_X_train.block(0, 0, conv_params[0].filter_size, conv_params[0].N*conv_params[0].N), conv_A3, conv_Ap3, z0_3, poolIdxX3, poolIdxY3);
	  
	  // Forward propagation 
	  std::vector<MyMatrix> Z3(n_layers-1);
	  std::vector<MyMatrix> A3(n_layers-2);
	  std::vector<MyMatrix> Ap3(n_layers-2);
	  fprop(false, 0., act_func, W, B, z0_3, Z3, A3, Ap3);
	  
	  // Compute the output and the error 
	  MyMatrix out3;
	  std::vector<MyMatrix> gradB(n_layers-1);
	  softmax(Z3[n_layers-2], out3);
	  gradB[n_layers-2] = out3 - one_hot.row(ex_id);
	  
	  // Backpropagation
	  bprop(W, Ap3, gradB);

	  // Backpropagation for conv layer
	  std::vector<MyMatrix> conv_gradB(conv_W.size());
	  MyMatrix layer_gradB = (gradB[0] * W[0].transpose());
	  MyMatrix pool_gradB;
	  layer2pool(1, pool_params[conv_W.size()-1].N, conv_params[conv_W.size()-1].n_filter, layer_gradB, pool_gradB);	  
	  convBprop(1, conv_params, pool_params, conv_W_T, conv_Ap3, pool_gradB, conv_gradB, poolIdxX3, poolIdxY3);

	  std::vector<MyMatrix> conv_update(conv_W.size());
	  std::vector<MyVector> conv_updateB(conv_W.size());
	  convUpdateTest(1,eta, conv_gradB, conv_A3, conv_X_train.block(0, 0, conv_params[0].filter_size, conv_params[0].N*conv_params[0].N), "", 0., conv_W, conv_B, conv_update, conv_updateB);
	  
	  double approx_dw = 0.0;
	  for(unsigned i = 0; i < all_delta_errors.rows(); i++){
	    approx_dw += all_delta_errors(i,Y_train(ex_id));
	  }
	  
	  if(abs(approx_dw) > 1e-10){
	    std::cout << "--------------------------------" << std::endl;
	    std::cout << "Conv Approx dw " << approx_dw << std::endl;
	    std::cout << "DW " << conv_update[l](i,j) << std::endl;
	    const double rel_error = abs(conv_update[l](i,j) - approx_dw) / (abs(approx_dw)+abs(conv_update[l](i,j)));
	    std::cout << "Rel_Error " << rel_error << std::endl;
	    
	    std::cout << "testing " << ex_id << " " << "conv " << l << " " << i << " " << j << " " << rel_error << " " << std::endl;
	    if(rel_error > REL_ERROR_THRESHOLD){
	      std::cout << "testing failed " << ex_id << " " << "conv " << i << " " << j << " " << approx_dw << " " << conv_update[l](i,j) << " " << rel_error << " " << std::endl;
	      assert(false);
	    }
	  }
	}
      }
    }
  
    std::cout << "Testing the convolutional bias" << std::endl;
    for(unsigned l = 0; l < conv_W.size(); l++){
      for(unsigned  i = 0; i < conv_B[l].size(); i++){
	
        // Change a parameter
        conv_B[l](i) += EPS;

	// Forward propagation for conv layer
	std::vector<std::vector<unsigned>> poolIdxX1(n_conv_layer);
	std::vector<std::vector<unsigned>> poolIdxY1(n_conv_layer);
	  
	MyMatrix z0_1;
	std::vector<MyMatrix> conv_A1(conv_W.size());
	std::vector<MyMatrix> conv_Ap1(conv_W.size());
	convFprop(1, conv_params, pool_params, act_func, conv_W, conv_B, conv_X_train.block(0, 0, conv_params[0].filter_size, conv_params[0].N*conv_params[0].N), conv_A1, conv_Ap1, z0_1, poolIdxX1, poolIdxY1);

	// Forward propagation
	std::vector<MyMatrix> Z1(n_layers-1);
	std::vector<MyMatrix> A1(n_layers-2);
	std::vector<MyMatrix> Ap1(n_layers-2);
	fprop(false, 0., act_func, W, B, z0_1, Z1, A1, Ap1);
	  
	// Compute the output and the error
	MyMatrix out1;
	softmax(Z1[n_layers-2], out1);
	MyMatrix error1 = out1 - one_hot.row(ex_id);
	
	// Change a parameter
	conv_B[l](i) -= 2.0 * EPS;

	// Forward propagation for conv layer
	std::vector<std::vector<unsigned>> poolIdxX2(n_conv_layer);
	std::vector<std::vector<unsigned>> poolIdxY2(n_conv_layer);
	
	MyMatrix z0_2;
	std::vector<MyMatrix> conv_A2(conv_W.size());
	std::vector<MyMatrix> conv_Ap2(conv_W.size());
	convFprop(1, conv_params, pool_params, act_func, conv_W, conv_B, conv_X_train.block(0, 0, conv_params[0].filter_size, conv_params[0].N*conv_params[0].N), conv_A2, conv_Ap2, z0_2, poolIdxX2, poolIdxY2);
	
	// Forward propagation
	std::vector<MyMatrix> Z2(n_layers-1);
	std::vector<MyMatrix> A2(n_layers-2);
	std::vector<MyMatrix> Ap2(n_layers-2);
	fprop(false, 0., act_func, W, B, z0_2, Z2, A2, Ap2);
	
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
	conv_B[l](i) += EPS;

	// Forward propagation for conv layer
	std::vector<std::vector<unsigned>> poolIdxX3(n_conv_layer);
	std::vector<std::vector<unsigned>> poolIdxY3(n_conv_layer);
	
	MyMatrix z0_3;
	std::vector<MyMatrix> conv_A3(conv_W.size());
	std::vector<MyMatrix> conv_Ap3(conv_W.size());
	convFprop(1, conv_params, pool_params, act_func, conv_W, conv_B, conv_X_train.block(0, 0, conv_params[0].filter_size, conv_params[0].N*conv_params[0].N), conv_A3, conv_Ap3, z0_3, poolIdxX3, poolIdxY3);
	
	// Forward propagation 
	std::vector<MyMatrix> Z3(n_layers-1);
	std::vector<MyMatrix> A3(n_layers-2);
	std::vector<MyMatrix> Ap3(n_layers-2);
	fprop(false, 0., act_func, W, B, z0_3, Z3, A3, Ap3);
	
	// Compute the output and the error 
	MyMatrix out3;
	std::vector<MyMatrix> gradB(n_layers-1);
	softmax(Z3[n_layers-2], out3);
	gradB[n_layers-2] = out3 - one_hot.row(ex_id);
	
	// Backpropagation
	bprop(W, Ap3, gradB);
	
	// Backpropagation for conv layer
	std::vector<MyMatrix> conv_gradB(conv_W.size());
	MyMatrix layer_gradB = (gradB[0] * W[0].transpose());
	MyMatrix pool_gradB;
	layer2pool(1, pool_params[conv_W.size()-1].N, conv_params[conv_W.size()-1].n_filter, layer_gradB, pool_gradB);	  
	convBprop(1, conv_params, pool_params, conv_W_T, conv_Ap3, pool_gradB, conv_gradB, poolIdxX3, poolIdxY3);
	
	std::vector<MyMatrix> conv_update(conv_W.size());
	std::vector<MyVector> conv_updateB(conv_W.size());
	convUpdateTest(1, eta, conv_gradB, conv_A3, conv_X_train.block(0, 0, conv_params[0].filter_size, conv_params[0].N*conv_params[0].N), "", 0., conv_W, conv_B, conv_update, conv_updateB);
	
	double approx_dw = 0.0;
	for(unsigned i = 0; i < all_delta_errors.rows(); i++){
	  approx_dw += all_delta_errors(i,Y_train(ex_id));
	}
	
	if(abs(approx_dw) > 1e-10){
	  std::cout << "--------------------------------" << std::endl;
	  std::cout << "Conv Bias Approx dw " << approx_dw << std::endl;
	  std::cout << "DW " << conv_updateB[l](i) << std::endl;
	  const double rel_error = abs(conv_updateB[l](i) - approx_dw) / (abs(approx_dw)+abs(conv_updateB[l](i)));
	  std::cout << "Rel_Error " << rel_error << std::endl;
	  
	  std::cout << "testing " << ex_id << " " << "convB " << l << " " << i << " " << " " << rel_error << " " << std::endl;
	  if(rel_error > REL_ERROR_THRESHOLD){
	    std::cout << "testing failed " << ex_id << " " << "convB " << i << " " << " " << approx_dw << " " << conv_updateB[l](i) << " " << rel_error << " " << std::endl;
   	    assert(false);
	  }
	}
      }
    }
  }
}
