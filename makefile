riemann: riemann.cpp nn_ops.hpp utils.hpp
	g++ -std=c++11 -O3 -o riemann riemann.cpp

convnet:
	g++ -std=c++11 -O3 convnet.cpp -o conv

convnet_test:
	g++ -std=c++11 -O3 conv_fdTest.cpp -o conv_fdTest

debug: riemann.cpp nn_ops.hpp utils.hpp
	g++ -std=c++11 -g -o riemann riemann.cpp

debug_sp: riemann_sp.cpp nn_ops_sp.hpp utils.hpp
	g++ -std=c++11 -g -o riemann_sp riemann_sp.cpp

riemann_sp: riemann_sp.cpp nn_ops_sp.hpp utils.hpp
	g++ -std=c++11 -O3 -o riemann_sp riemann_sp.cpp

test: fdTest.cpp nn_ops.hpp utils.hpp
	g++ -std=c++11 -O3 -o fdTest fdTest.cpp

conv_test: conv_fdTest.cpp nn_ops.hpp utils.hpp
	g++ -std=c++11 -O3 -o conv_fdTest conv_fdTest.cpp

test_sp: fdSpTest.cpp nn_ops.hpp utils.hpp
	g++ -std=c++11 -O3 -o fdSpTest fdSpTest.cpp

clean: 
	rm riemann riemann_sp fdTest fdSpTest *~
