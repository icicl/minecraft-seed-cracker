nvcc -O3 -Xcompiler -fPIC -shared crack3.cu -o crack.so
python3 main.py
