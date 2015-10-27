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
 * \file utils.hpp
 * \brief Implementation of auxiliary functions required for dataset loading and outputting 
 * \author Gaetan Marceau Caron & Yann Ollivier
 * \version 1.0
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <map>
#include <time.h>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace Eigen;

const std::string CIFAR_DIR = "./cifar-10-batches-bin/";
const std::string MNIST_DIR = "./mnist/";

const unsigned N_ROW = 10000;
const unsigned N_CHANNEL = 1;
const unsigned IMG_WIDTH = 28;
const unsigned IMG_HEIGHT = 28;

const unsigned N_PIXEL = IMG_WIDTH * IMG_HEIGHT;

// Typename declaration
using MySpMatrix=Eigen::SparseMatrix<double,ColMajor>;
using MyMatrix=Eigen::Matrix<double,Dynamic,Dynamic,ColMajor>;
using SpEntry=Eigen::Triplet<double>;
using MyVector=Eigen::Matrix<double,Dynamic, 1, ColMajor>;
using ActivationFunction = std::function<void (const MyMatrix&,MyMatrix&,MyMatrix&)>;
using ForwardFunction = std::function<void (const ActivationFunction&,
											const std::vector<MyMatrix>&, 
											const std::vector<MyVector>&,
											const MyMatrix&,
											std::vector<MyMatrix>&,
											std::vector<MyMatrix>&,
											std::vector<MyMatrix>&)>;

using namespace std::placeholders;

// Random Generator
std::mt19937 gen;

struct Params{
  
	std::string algo;
	std::vector<unsigned> nn_arch;
	unsigned n_params;
	std::string act_func_name;
	double ratio_train;
	bool schedule_flag;
	
	double eta;
	double eta_min;
	bool adaptive_flag;
	bool dropout_flag;
	double dropout_prob;
	bool dropout_eval;

	unsigned train_minibatch_size;
	unsigned valid_minibatch_size;
	unsigned minibatch_size;
	double metric_gamma;
	bool init_metric_id;
	unsigned n_epoch;
	double matrix_reg;
	unsigned sparsity;
	double imp_sampling_ratio;
	bool invert_pixel;
	
	unsigned img_width;
	unsigned img_height;
	unsigned img_depth;
  
	unsigned conv_Hf1;
	unsigned conv_Hf2;
	unsigned conv_padding;
	unsigned conv_n_filter;
	unsigned conv_stride1;
	unsigned conv_stride2;
  
	std::string regularizer;
	double lambda;

	bool minilog_flag;
	std::string dir_path;
	std::string exp_desc;
	std::string file_path;
	unsigned run_id;

	unsigned seed;
};

// Time utilities
auto t=clock();
void printtime(){
	cout<<(float)(clock()-t) / CLOCKS_PER_SEC<<endl;
	t=clock();
}

auto starting_time=clock();
double gettime(){
	double curr_time = (double)(clock()-starting_time) / CLOCKS_PER_SEC;
	return curr_time;
}

/**
 *  \brief Implementation of the change of encoding
 *  \param i the variable to transform
 *  \return the transformed variable
 */
int ReverseInt (const int i){
	unsigned char ch1, ch2, ch3, ch4;
	ch1=i&255;
	ch2=(i>>8)&255;
	ch3=(i>>16)&255;
	ch4=(i>>24)&255;
	return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+(int)ch4;
}

/**
 *  \brief Parse a sequence of number given as a string
 *  \param seq_str the string to parse
 *  \param seq the vector of numbers
 */
void parseYAMLSeq(const std::string seq_str,
				  std::vector<unsigned> &seq){

	std::string nstr = seq_str.substr(1,seq_str.length()-2);
	std::cout << "substring " << nstr << std::endl;
	size_t idx = nstr.find(',');
	while(idx != std::string::npos){
		std::string value = nstr.substr(0,idx);
		seq.push_back(std::stoi(value));

		nstr = nstr.substr(idx+1);
		std::cout << "seq " << value << " " << nstr << std::endl;
		idx = nstr.find(',');
	}
	seq.push_back(std::stoi(nstr));
  
}

/**
 *  \brief Parse the arguments from the command line
 *  \param argc the number of arguments to parse
 *  \param argv the vector of arguments given as strings
 *  \param args a map containing pairs of key/value strings
 */
void readArgs(const int argc, 
			  char *argv[],
			  std::map<std::string, std::string> &args){
  
	unsigned i = 1;
	while(i < argc){
		std::string param(argv[i]);
		if(param.at(0)=='-'){
			param.erase(0,1);
			std::string value(argv[++i]);
			args[param] = value;
			std::cout << param << " " << value << std::endl;
		}
		i++;
	}
}

/**
 *  \brief Fill the params holder structure with a dictionnary of key/value strings
 *  \param args a map containing pairs of key/value strings
 *  \param params a param holder structure
 */
void setParamsFromArgs(std::map<std::string, std::string> &args,
					   Params& params){
	
	if(args.find("algo")!=args.end()){
		params.algo = args["algo"];
	}
	if(args.find("eta")!=args.end()){
		params.eta = std::stod(args["eta"]);
	}
	if(args.find("metric_gamma")!=args.end()){
		params.metric_gamma = std::stod(args["metric_gamma"]);
	}
	if(args.find("matrix_reg") != args.end()){
		params.matrix_reg = std::stod(args["matrix_reg"]);
	}
	if(args.find("imp_sampling") != args.end()){
		params.imp_sampling_ratio = std::stod(args["imp_sampling"]);
	}
	if(args.find("dropout_flag")!=args.end()){
		params.dropout_flag = std::stoi(args["dropout_flag"]);
	}
	if(args.find("regularizer")!=args.end()){
		params.regularizer = args["regularizer"];
	}
	if(args.find("lambda")!=args.end()){
		params.lambda = std::stod(args["lambda"]);
	}
	if(args.find("dropout_prob")!=args.end()){
		params.dropout_prob = std::stod(args["dropout_prob"]);
	}
	if(args.find("dropout_eval")!=args.end()){
		params.dropout_eval = std::stod(args["dropout_eval"]);
	}
	if(args.find("nn_arch")!=args.end()){
		params.nn_arch.clear();
		parseYAMLSeq(args["nn_arch"], params.nn_arch);
	}
	if(args.find("sparsity") != args.end()){
		params.sparsity = std::stoi(args["sparsity"]);
	}
	if(args.find("minibatch_size") != args.end()){
		params.minibatch_size = std::stoi(args["minibatch_size"]);
	}
	if(args.find("adaptive_flag")!=args.end()){
		params.adaptive_flag = std::stoi(args["adaptive_flag"]);
	}
	if(args.find("schedule_flag")!=args.end()){
		params.schedule_flag = std::stoi(args["schedule_flag"]);
	}
	if(args.find("invert_pixel")!=args.end()) {
		params.invert_pixel = std::stoi(args["invert_pixel"]);
	}
	if(args.find("run_id")!=args.end()) {
		params.run_id = std::stoi(args["run_id"]);
		params.seed = (unsigned int)time(0) + params.run_id * 9973;
	}
	if(args.find("act_func") != args.end()){
		params.act_func_name = args["act_func"];
	}  
	if(args.find("minilog_flag")!=args.end()) {
		params.minilog_flag = std::stoi(args["minilog_flag"]);
	}
	if(args.find("init_metric_id")!=args.end()){
		params.init_metric_id = std::stoi(args["init_metric_id"]);
	}
	if(args.find("seed")!=args.end()){
		params.seed = std::stoi(args["seed"]);
	}
	if(args.find("n_epoch")!=args.end()){
		params.n_epoch = std::stoi(args["n_epoch"]);
	}
	if(args.find("dir_path") != args.end()){
		params.dir_path = args["dir_path"];
	}
	if(args.find("exp_desc") != args.end()){
		params.exp_desc = args["exp_desc"];
	}

	std::ostringstream strs;
	strs << params.dir_path << params.algo << "_eta" << params.eta << "_gam" << params.metric_gamma << "_reg" << params.matrix_reg;
  
	params.file_path = strs.str(); 

	if(params.init_metric_id){
		params.file_path += "_initId";
	}

	if(params.exp_desc!="")
		params.file_path += "_" + params.exp_desc;

	if(params.run_id>0)
		params.file_path += "_run" + std::to_string(params.run_id);

	std::cout << params.file_path << std::endl;
  
}

/**
 *  \brief Set the default parameters according to the chosen algorithm
 *  \param algo the name of the algorithm
 *  \param args a map containing pairs of key/value strings
 *  \param params a param holder structure
 */
void setParams(const std::string algo,
			   std::map<std::string, std::string> args, 
			   Params& params){

	params.algo = algo;
	params.ratio_train = 0.8333333333;
	params.train_minibatch_size = 500;
	params.valid_minibatch_size = 500;
	params.minibatch_size = 500;
	params.n_epoch = 400;
	params.seed = (unsigned int)time(0);
	params.adaptive_flag = false;
	params.schedule_flag = false;
	params.dropout_flag = false;
	params.dropout_prob = 0.5;
	params.dropout_eval = false;
	params.minilog_flag = false;
	params.dir_path = "./results/";
	params.run_id = 0;
	params.regularizer = "";
	params.lambda = 1e-4;
	params.init_metric_id = false;
	params.sparsity = 0;
	params.eta_min = 1e-4;
	params.imp_sampling_ratio = -1.0;
	params.invert_pixel = false;

	// Image parameters
	params.img_width = IMG_WIDTH;
	params.img_height = IMG_HEIGHT;
	params.img_depth = N_CHANNEL;
  
	// Convolutional Parameters
	params.conv_n_filter = 16;
	params.conv_Hf1 = 3;
	params.conv_Hf2 = 2;
	params.conv_stride1 = 1;
	params.conv_stride2 = 2;
	params.conv_padding = 0;
  
	const unsigned nn_arch_conf[] = {100};
	std::vector<unsigned> nn_arch(nn_arch_conf, nn_arch_conf + sizeof(nn_arch_conf)/sizeof(unsigned));
	params.nn_arch = nn_arch;
	params.act_func_name = "relu";

	if(algo=="bprop"){
		params.eta = 1.0;
		params.matrix_reg = 0.;
		params.metric_gamma = 0.;
	}else{
		params.matrix_reg = 1e-8;
    
		if(algo=="adagrad"){
			params.eta = 0.01;
		}else if(algo=="qdop"){
			params.eta = 0.001;
			params.metric_gamma = 0.01;
		}else if(algo=="qdMCNat"){
			params.eta = 0.01;
			params.metric_gamma = 0.1;
		}else if(algo=="qdNat"){
			params.eta = 0.0001;
			params.metric_gamma = 0.01;
		}else if(algo=="qdbpm"){
			params.eta = 0.0001;
			params.metric_gamma = 0.01;
		}else if(algo=="gaussNewton"){
			params.eta = 0.0001;
			params.metric_gamma = 0.01;
		}else if(algo=="dop"){
			params.eta = 0.0001;
			params.metric_gamma = 0.01;
		}else if(algo=="rmsprop"){
			params.eta = 0.0001;
			params.metric_gamma = 0.01;
		}
	}
	setParamsFromArgs(args, params);

}

/**
 *  \brief Read the arguments from a YAML file
 *  \param filename the path of the YAML file
 *  \param config a map containing pairs of key/value strings
 */
void loadYAML(const std::string filename,
			  std::map<std::string, std::string> &config){

	ifstream yaml_file(filename.c_str(), ios::in);
  
	std::string line;

	while(std::getline (yaml_file,line)){
		size_t idx = line.find(':');
		std::string key = line.substr(0,idx);
		std::string value = line.substr(idx+2);
    
		config[key] = value;
	}

}

/**
 *  \brief Read the arguments and save them into the parameters holder structure
 *  \param filename the path of the YAML file
 *  \param args a map containing pairs of key/value strings
 *  \param params a param holder structure
 */
void setParamsFromFile(const std::string filename,
					   std::map<std::string, std::string> &args,
					   Params& params){

	std::map<std::string, std::string> config;
	loadYAML(filename, config);
  
	setParams(config["algo"], args, params);
	if(config.find("eta") != config.end()){
		params.eta = std::stod(config["eta"]);
	}
	if(config.find("nn_arch") != config.end()){
		params.nn_arch.clear();
		parseYAMLSeq(config["nn_arch"], params.nn_arch);
	}
	if(config.find("sparsity") != config.end()){
		params.sparsity = std::stoi(config["sparsity"]);
	}
	if(config.find("adaptive_flag") != config.end()){
		params.adaptive_flag = std::stoi(config["adaptive_flag"]);
	}
	if(config.find("schedule_flag") != config.end()){
		params.schedule_flag = std::stoi(config["schedule_flag"]);
	}
	if(config.find("act_func") != config.end()){
		params.act_func_name = config["act_func"];
	}
	if(config.find("minibatch_size") != config.end()){
		params.minibatch_size = std::stoi(config["minibatch_size"]);
	}
	if(config.find("metric_gamma") != config.end()){
		params.metric_gamma = std::stod(config["metric_gamma"]);
	}
	if(config.find("n_epoch") != config.end()){
		params.n_epoch = std::stoi(config["n_epoch"]);
	}
	if(config.find("matrix_reg") != config.end()){
		params.matrix_reg = std::stod(config["matrix_reg"]);
	}
	if(config.find("init_metric_id") != config.end()){
		params.init_metric_id = std::stoi(config["init_metric_id"]);
	}  
	if(config.find("dropout_flag") != config.end()){
		params.dropout_flag = std::stoi(config["dropout_flag"]);
	}
	if(config.find("dropout_prob") != config.end()){
		params.dropout_prob = std::stod(config["dropout_prob"]);
	}
	if(config.find("minilog_flag") != config.end()){
		params.minilog_flag = std::stoi(config["minilog_flag"]);
	}
	if(config.find("dir_path") != config.end()){
		params.dir_path = config["dir_path"];
	}
	if(config.find("seed") != config.end()){
		params.seed = std::stoi(config["seed"]);
	}

	setParamsFromArgs(args, params);
}

/**
 *  \brief print the description of the parameters
 *  \param params a param holder structure
 *  \param logger the output stream to print on
 */
void printDesc(Params params,
			   ostream* logger){
	*logger << "Description of the experiment (C++)" << std::endl;
	*logger << "----------" << std::endl;
	if(params.sparsity==0)
		*logger << "Learning algorithm: Full " + params.algo << std::endl;
	else
		*logger << "Learning algorithm: Sparse " + params.algo << std::endl;
	*logger << "Initial step-size: " << params.eta << std::endl;
	*logger << "Network Architecture: ";
	for(unsigned i = 0; i < params.nn_arch.size(); i++){
		*logger << params.nn_arch[i] << " ";
	}
	*logger << std::endl;
	if(params.sparsity>0)
		*logger << "Sparsity: " << params.sparsity << std::endl;
	*logger << "Number of parameters: " << params.n_params << std::endl;
	*logger << "Activation: " << params.act_func_name << std::endl;
	*logger << "Minibatch size: " << params.minibatch_size << std::endl;
	if(params.dropout_flag){
		*logger << "Regularization with dropout (p=" << params.dropout_prob << ")" << std::endl;
		if(params.dropout_eval){
			*logger << "Evaluation with dropout " << std::endl;
		}
	}
	if(params.regularizer!=""){
		*logger << "Regularization with " << params.regularizer << "(lambda=" << params.lambda << ")" << std::endl;
	}
	*logger << "Metric update coefficient: " << params.metric_gamma << std::endl;
	*logger << "Initialize metric with Id: " << params.init_metric_id << std::endl;
	*logger << "Matrix regularization: " << params.matrix_reg << std::endl;
	*logger << "Seed: " << std::fixed << params.seed << std::endl;
	if(params.imp_sampling_ratio>0.){
		assert(params.algo=="qdMCNat");
		*logger << "Importance Sampling ratio (QDMCNat): " << std::fixed << params.imp_sampling_ratio << std::endl;
	}

	*logger << "----------" << std::endl;
}

/**
 *  \brief read the MNIST images
 *  \param filename the filename of the MNIST data file
 *  \param images the design matrix of the images
 *  \param invert_pixel a flag to invert the pixel of the images
 */
void readMNIST(const string filename, 
			   MyMatrix &images,
			   bool invert_pixel){
  
	ifstream file(filename.c_str(), ios::binary);
	if (file.is_open()){
		unsigned n_training = 0;
		int magic_number = 0;
		int n_rows = 0;
		int n_cols = 0;

		file.read((char*) &magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*) &n_training,sizeof(n_training));
		n_training = ReverseInt(n_training);
		file.read((char*) &n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*) &n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);

		images.resize(n_training, n_rows*n_cols);
    
		for(int i = 0; i < n_training; ++i){
			for(int r = 0; r < n_rows; ++r){
				for(int c = 0; c < n_cols; ++c){
					unsigned char temp = 0;
					file.read((char*) &temp, sizeof(temp));
					if(!invert_pixel)
						images(i,r*n_cols+c) = (double) temp /255.;
					else{
						images(i,r*n_cols+c) = (1.0 - (double) temp /255.);
					}
				}
			}
		}
	}
}

/**
 *  \brief read the MNIST labels
 *  \param filename the filename of the MNIST data file
 *  \param labels the labels associated to the images (same order than previous function)
 */
void readMNISTLabel(const string filename,
					MyVector &labels){
  
	ifstream file (filename.c_str(), ios::binary);
	if (file.is_open()){
		int magic_number = 0;
		int n_training = 0;
		int n_rows = 0;
		int n_cols = 0;
    
		file.read((char*) &magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*) &n_training,sizeof(n_training));
		n_training = ReverseInt(n_training);
    
		labels.resize(n_training);
    
		for(int i = 0; i < n_training; ++i){
			unsigned char temp = 0;
			file.read((char*) &temp, sizeof(temp));
			labels(i)= (unsigned)temp;
		}
	}
}

/**
 *  \brief read the CIFAR-10 images
 *  \param images the design matrix of the images
 *  \param labels the labels associated to the images (same order than previous function)
 */
int readCIFAR10(const std::string filepath,
				MyMatrix& images,
				MyVector& labels){

	images.resize(N_ROW, N_CHANNEL*N_PIXEL);
	labels.resize(N_ROW);
  
	ifstream file(filepath.c_str(), ios::binary);
	if(file.is_open()){
		for(unsigned i = 0; i < N_ROW; i++){
			unsigned char label;
			file.read((char*) &label, sizeof(label));
			labels[i] = label;
			for(unsigned j = 0; j < N_CHANNEL; j++){
				for(unsigned k = 0; k < N_PIXEL; k++){
					unsigned char pixel;
					file.read((char*) &pixel, sizeof(char));
					images(i,j*N_PIXEL+k) = pixel;
				}
			}
		}
	}else{
		std::cout << "Error while loading dataset: " << filepath << std::endl;
	}
}

void loadLightCIFAR10(const double ratio_train,
					  MyMatrix &X_train,
					  MyMatrix &X_valid,
					  MyVector &Y_train,
					  MyVector &Y_valid){
  
	// Load the dataset
	MyMatrix images;
	MyVector labels;
	readCIFAR10(CIFAR_DIR+"data_batch_1.bin",images,labels);

	const unsigned n_images = images.rows();
  
	// Split the dataset into training and validation
	const unsigned idx_end = n_images * ratio_train; 
	X_train = images.topRows(idx_end);
	X_valid = images.bottomRows(n_images-idx_end);
  
	Y_train = labels.topRows(idx_end);
	Y_valid = labels.bottomRows(n_images-idx_end);
}

void loadCIFAR10(const double ratio_train,
				 MyMatrix &X_train,
				 MyMatrix &X_valid,
				 MyVector &Y_train,
				 MyVector &Y_valid){
  
	// Load the dataset
	MyMatrix images1;
	MyVector labels1;
	readCIFAR10(CIFAR_DIR+"data_batch_1.bin",images1,labels1);

	MyMatrix images2;
	MyVector labels2;
	readCIFAR10(CIFAR_DIR+"data_batch_2.bin",images2,labels2);

	MyMatrix images3;
	MyVector labels3;
	readCIFAR10(CIFAR_DIR+"data_batch_3.bin",images3,labels3);

	MyMatrix images4;
	MyVector labels4;
	readCIFAR10(CIFAR_DIR+"data_batch_4.bin",images4,labels4);

	MyMatrix images5;
	MyVector labels5;
	readCIFAR10(CIFAR_DIR+"data_batch_5.bin",images5,labels5);

	const unsigned n_images = images1.rows() + images2.rows() + images3.rows() + images4.rows() + images5.rows();
	MyMatrix images(n_images, images1.cols());
	MyVector labels(n_images);

	images << images1, images2, images3, images4, images5;
	labels << labels1, labels2, labels3, labels4, labels5;
  
  
	// Split the dataset into training and validation
	const unsigned idx_end = n_images * ratio_train; 
	X_train = images.topRows(idx_end);
	X_valid = images.bottomRows(n_images-idx_end);
  
	Y_train = labels.topRows(idx_end);
	Y_valid = labels.bottomRows(n_images-idx_end);
}

void shuffleDatabase(MyMatrix &data,
					 MyVector &labels,
					 unsigned n_examples=0){

	// Shuffle the examples
	PermutationMatrix<Dynamic,Dynamic> perm(labels.size());
	perm.setIdentity();
	std::random_shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
	data = perm * data; // permute rows
	labels = perm * labels; // permute rows

	if(n_examples>0){
		MyMatrix data2 = data.topRows(n_examples);
		MyVector labels2 = labels.head(n_examples);
		data = data2;
		labels = labels2;
	}
}

void loadMnist(const Params &params,
			   MyMatrix &X_train,
			   MyMatrix &X_valid,
			   MyVector &Y_train,
			   MyVector &Y_valid){
  
	// Load the dataset
	MyMatrix images;
	MyVector labels;
	readMNIST(MNIST_DIR+"train-images.idx3-ubyte", images, params.invert_pixel);
	readMNISTLabel(MNIST_DIR+"train-labels.idx1-ubyte", labels);
  
	// Split the dataset into training and validation
	const unsigned n_images = images.rows();
	const unsigned idx_end = n_images * params.ratio_train; 
	X_train = images.topRows(idx_end);
	X_valid = images.bottomRows(n_images-idx_end);
  
	Y_train = labels.topRows(idx_end);
	Y_valid = labels.bottomRows(n_images-idx_end);
  
}

void getMiniBatch(const unsigned j,
				  const unsigned batch_size,
				  const MyMatrix &X, 
				  const VectorXd &Y,
				  const MyMatrix &one_hot,
				  unsigned &curr_batch_size,
				  MyMatrix &X_batch, 
				  MyMatrix &one_hot_batch){

	X_batch.resize(batch_size,X.cols());
  
	const unsigned n_training = X.rows();
	const unsigned idx_begin = j * batch_size;
	const unsigned idx_end = std::min((j+1) * batch_size, n_training);
	curr_batch_size = idx_end - idx_begin;
  
	X_batch = X.middleRows(idx_begin, curr_batch_size);
	one_hot_batch = one_hot.middleRows(idx_begin, curr_batch_size);
}

void labels2oneHot(const MyMatrix &labels,
				   MyMatrix &one_hot){
	for(unsigned i = 0; i < labels.size(); i++){
		one_hot(i, labels(i)) = 1.;
	}
}

void writeNetwork(const std::string filename,
				  std::vector<MyMatrix> &W,
				  std::vector<MyVector> &B){

	ofstream file(filename.c_str(), ios::binary);
	if(file.is_open()){
		unsigned n_layer = W.size();
		file.write((char*) &n_layer, sizeof(n_layer));
		for(unsigned i = 0; i < n_layer; i++){
			unsigned n_rows = W[i].rows();
			unsigned n_cols = W[i].cols();
			file.write((char*) &n_rows, sizeof(n_rows));
			file.write((char*) &n_cols, sizeof(n_cols));

			std::cout << n_rows << " " << n_cols << " " << B[i].size() << std::endl;

			// Write the weight matrix
			for(unsigned j = 0; j < n_rows; j++){
				for(unsigned k = 0; k < n_cols; k++){
					file.write((char*) &W[i](j,k), sizeof(W[i](j,k)));
				}
			}

			// Write the bias vector
			for(unsigned j = 0; j < n_cols; j++){
				file.write((char*) &B[i](j), sizeof(B[i](j)));
			}
		}
		file.close();
	}
}

int loadNetwork(const std::string filename,
				std::vector<MyMatrix> &W,
				std::vector<MyVector> &B){

	ifstream file(filename.c_str(), ios::binary);
	unsigned n_params = 0;
	if(file.is_open()){
		unsigned n_layer = 0;
		file.read((char*) &n_layer, sizeof(n_layer));
		W.resize(n_layer);
		std::cout << "N layer " << (int)n_layer << " " << sizeof(n_layer) << std::endl;
		for(unsigned i = 0; i < n_layer; i++){
			unsigned n_rows,n_cols;
			file.read((char*) &n_rows, sizeof(n_rows));
			file.read((char*) &n_cols, sizeof(n_cols));
			W[i].resize(n_rows,n_cols);
			B[i].resize(n_cols);
			std::cout << n_rows << " " << n_cols << std::endl;
      
			// Read the weight matrix
			for(unsigned j = 0; j < n_rows; j++){
				for(unsigned k = 0; k < n_cols; k++){
					double data = 0.;
					file.read((char*) &data, sizeof(data));
					W[i](j,k) = data;
					n_params++;
				}
			}

			// Read the bias vector
			for(unsigned j = 0; j < n_cols; j++){
				double data = 0.;
				file.read((char*) &data, sizeof(data));
				B[i](j) = data;
				n_params++;
			}
		}
	}
	return n_params;
}

void createLogDir(const std::string dir){
	struct stat st = {0};
	if (stat(dir.c_str(), &st) == -1) {
		mkdir(dir.c_str(), 0700);
	}
}

void getOutput(const std::string path, ostream* &out){
  
	if(path!=""){
		ofstream* file_out = new std::ofstream(path, ios::out);
		if (file_out->is_open()){
			out = file_out;
			return;
		}else{
			std::cout << "error" << std::endl;
		}
	}
	out = &std::cout;
  
}

void monitor(const std::vector<MyMatrix> &A_vec){

	for(unsigned l = 0; l < A_vec.size(); l++){
		//std::cout << A_vec[l].rows() << " " << A_vec[l].cols() << std::endl;
		MyVector means = A_vec[l].colwise().mean();

		double mean = means.mean();
		double variance = means.array().square().mean() - pow(mean,2);
		std::cout << "Layer " << l << " " << mean << " " << variance << std::endl;
	}
	std::cout << "-----" << std::endl;
}


#endif
