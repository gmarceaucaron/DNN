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
 * \brief Implementation of auxiliary functions for dataset loading, parameter handling  and outputting 
 * \author Gaetan Marceau Caron & Yann Ollivier
 * \version 1.0
 */

#ifndef UTILS_CONV_HPP_
#define UTILS_CONV_HPP_


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

#include "utils.hpp"

struct ConvLayerParams{
  unsigned Hf;
  unsigned padding;
  unsigned n_filter;
  unsigned stride;

  unsigned filter_size;
  unsigned N;

};

struct PoolLayerParams{
  unsigned Hf;
  unsigned padding;
  unsigned stride;

  unsigned filter_size;
  unsigned N;
};

void printConvDesc(Params params,
		   std::vector<ConvLayerParams> conv_params,
		   std::vector<PoolLayerParams> pool_params,
		   ostream* logger){

  *logger << "image information" << std::endl;
  *logger << "height " << params.img_height << std::endl;
  *logger << "width " << params.img_width << std::endl;
  *logger << "depth " << params.img_depth << std::endl;

  *logger << "filter information" << std::endl;
  for(unsigned l = 0; l < conv_params.size(); l++){
    *logger << "Conv Hf " << conv_params[l].Hf << std::endl;
    *logger << "Conv stride " << conv_params[l].stride << std::endl;
    *logger << "Conv padding " << conv_params[l].padding << std::endl;
    
    *logger << "Pool hf " << pool_params[l].Hf << std::endl;
    *logger << "Pool stride " << pool_params[l].stride << std::endl;
    
    *logger << "n filter " << conv_params[l].n_filter << std::endl;
    *logger << "filter size " << conv_params[l].filter_size << std::endl;
    *logger << "conv N " << conv_params[l].N << std::endl;
    *logger << "pool N " << pool_params[l].N << std::endl << std::endl;
  }
}

void data2conv(const MyMatrix &img_batch,
	       const unsigned n_img,
	       const unsigned img_width,
	       const unsigned img_height,
	       const unsigned img_n_channel,
	       const unsigned conv_N,
	       const unsigned conv_Hf,
	       const unsigned conv_padding,
	       const unsigned conv_stride,
	       MyMatrix &X_batch){

  X_batch.resize(img_n_channel*conv_Hf*conv_Hf, conv_N*conv_N*n_img);
  for(unsigned i = 0; i < img_batch.rows(); i++){
    for(unsigned j = 0; j < img_n_channel; j++){
      for(unsigned no = 0; no < conv_N; no++){
	for(unsigned ni = 0; ni < conv_N; ni++){
	  for(unsigned ko = 0; ko < conv_Hf; ko++){
	    for(unsigned ki = 0; ki < conv_Hf; ki++){
	      const unsigned pixel_idx = ki+ko*(img_width+2*conv_padding) + ni*conv_stride + no*conv_stride*img_width+j*img_width*img_height;
	      const unsigned idx1 = j*conv_Hf*conv_Hf+ko*conv_Hf+ki;
	      const unsigned idx2 = i*conv_N*conv_N+no*conv_N+ni;
	      X_batch(idx1,idx2) = img_batch(i,pixel_idx);
	    }
	  }
	}
      }
    }
  }  
}

void data2conv(const MyMatrix &img_batch,
	       const unsigned n_img,
	       const Params& params,
	       const ConvLayerParams &conv_params,
	       MyMatrix &X_batch){

  X_batch.resize(params.img_depth*conv_params.Hf*conv_params.Hf, conv_params.N*conv_params.N*n_img);
  for(unsigned i = 0; i < img_batch.rows(); i++){
    for(unsigned j = 0; j < params.img_depth; j++){
      for(unsigned no = 0; no < conv_params.N; no++){
	for(unsigned ni = 0; ni < conv_params.N; ni++){
	  for(unsigned ko = 0; ko < conv_params.Hf; ko++){
	    for(unsigned ki = 0; ki < conv_params.Hf; ki++){
	      const unsigned pixel_idx = ki+ko*(params.img_width+2*conv_params.padding) + ni*conv_params.stride + no*conv_params.stride*params.img_width+j*params.img_width*params.img_height;
	      const unsigned idx1 = j*conv_params.Hf*conv_params.Hf+ko*conv_params.Hf+ki;
	      const unsigned idx2 = i*conv_params.N*conv_params.N+no*conv_params.N+ni;
	      X_batch(idx1,idx2) = img_batch(i,pixel_idx);
	    }
	  }
	}
      }
    }
  }  
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
	      const unsigned idx1 = j*F*F+ko*F+ki;
	      const unsigned idx2 = no*conv_N2+ni+i*conv_N2*conv_N2;
	      conv_matrix(idx1,idx2) = conv_layer(j,idx);
	    }
	  }
	}
      }
    }
  }
}

void getMiniBatch(const unsigned j,
		  const unsigned minibatch_size,
		  const MyMatrix &X, 
		  const MyMatrix &one_hot,
		  const Params &params,
		  const ConvLayerParams &conv_params,
		  unsigned &n_img,
		  MyMatrix &X_batch, 
		  MyMatrix &one_hot_batch){

  X_batch.resize(minibatch_size,X.cols());
  
  const unsigned n_training = X.rows();
  const unsigned idx_begin = j * minibatch_size;
  const unsigned idx_end = std::min((j+1) * minibatch_size, n_training);
  n_img = idx_end - idx_begin;
  
  MyMatrix img_batch = X.middleRows(idx_begin, n_img);
  one_hot_batch = one_hot.middleRows(idx_begin, n_img);

  data2conv(img_batch, n_img, params, conv_params, X_batch);
}

void getMiniBatch(const unsigned j,
		  const unsigned minibatch_size,
		  const MyMatrix &X, 
		  const MyVector &Y,
		  const Params &params,
		  const ConvLayerParams &conv_params,
		  unsigned &n_img,
		  MyMatrix &X_batch, 
		  MyVector &Y_batch){
  X_batch.resize(minibatch_size,X.cols());
  
  const unsigned n_training = X.rows();
  const unsigned idx_begin = j * minibatch_size;
  const unsigned idx_end = std::min((j+1) * minibatch_size, n_training);
  n_img = idx_end - idx_begin;
  
  MyMatrix img_batch = X.middleRows(idx_begin, n_img);
  Y_batch = Y.middleRows(idx_begin, n_img);

  data2conv(img_batch, n_img, params, conv_params, X_batch);
}

void getMiniBatch(const unsigned j,
		  const unsigned batch_size,
		  const MyMatrix &X, 
		  const VectorXd &Y,
		  const MyMatrix &one_hot,
		  const unsigned img_width,
		  const unsigned img_height,
		  const unsigned img_depth,
		  const unsigned conv_N,
		  const unsigned conv_Hf,
		  const unsigned conv_padding,
		  const unsigned conv_stride,
		  unsigned &n_img,
		  MyMatrix &X_batch, 
		  MyMatrix &one_hot_batch){

  X_batch.resize(batch_size,X.cols());
  
  const unsigned n_training = X.rows();
  const unsigned idx_begin = j * batch_size;
  const unsigned idx_end = std::min((j+1) * batch_size, n_training);
  n_img = idx_end - idx_begin;
  
  MyMatrix img_batch = X.middleRows(idx_begin, n_img);
  one_hot_batch = one_hot.middleRows(idx_begin, n_img);

  data2conv(img_batch, n_img, img_width, img_height, img_depth, conv_N, conv_Hf, conv_padding, conv_stride, X_batch);
}

void pool2conv(const unsigned n_img,
	       const unsigned n_filter,
	       const unsigned conv_N1,
	       const unsigned conv_N2,
	       const std::vector<unsigned> &pool_idx_x,
	       const std::vector<unsigned> &pool_idx_y,
	       const MyMatrix &pool_z,
	       MyMatrix &conv_z){

  conv_z.setZero(n_filter, conv_N1*conv_N1*n_img);
  unsigned kk = 0;
  for(unsigned i = 0; i < n_img; i++){
    for(unsigned j = 0; j < n_filter; j++){
      for(unsigned k = 0; k < conv_N2; k++){
	for(unsigned l = 0; l < conv_N2; l++){
	  unsigned idx = i*conv_N2*conv_N2+k*conv_N2+l;
	  conv_z(pool_idx_x[kk],pool_idx_y[kk]) = pool_z(j,idx);
	  kk++;
	}
      }
    }
  }
}

void conv2Layer(const unsigned n_img,
		const unsigned N, 
		const MyMatrix &conv_obj,
		MyMatrix &layer){

  const unsigned n_filter = conv_obj.rows();
  const unsigned filter_size = conv_obj.cols()/n_img; 
  
  layer.resize(n_filter*filter_size,n_img);
  
  for(unsigned i = 0; i < n_img; i++){
    MyMatrix sub_conv = conv_obj.block(0, i*N*N, n_filter, N*N).transpose();
    layer.col(i) = Map<MyVector>(sub_conv.data(), sub_conv.cols()*sub_conv.rows());
  }
  layer.transposeInPlace();
}

void layer2pool(const unsigned n_img,
		const unsigned N,
		const unsigned n_filter,
		const MyMatrix &layer,
		MyMatrix &conv_obj){
  const unsigned filter_size = layer.cols()/n_filter;
  conv_obj.resize(n_filter,filter_size*n_img);
  for(unsigned i = 0; i < n_img; i++){
    for(unsigned j = 0; j < n_filter; j++){
      for(unsigned k = 0; k < filter_size; k++){
	conv_obj(j,i*filter_size+k) = layer(i,j*filter_size+k);
      }
    }
  }
}

#endif
