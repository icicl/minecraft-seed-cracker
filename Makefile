crack.so: crack3.cu
	nvcc -O3 -Xcompiler -fPIC -shared crack3.cu -o crack.so
