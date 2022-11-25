cd src
nvcc -c -o deform_conv_2d_cuda_kernel.cu.so deform_conv_2d_cuda_kernel.cu -x cu -Xcompiler -fPIC -std=c++11
nvcc -c -o deform_conv_3d_cuda_kernel.cu.so deform_conv_3d_cuda_kernel.cu -x cu -Xcompiler -fPIC -std=c++11

cd cuda

# compile modulated deform conv
nvcc -c -o modulated_deform_im2col_2d_cuda.cu.so modulated_deform_im2col_2d_cuda.cu -x cu -Xcompiler -fPIC
nvcc -c -o modulated_deform_im2col_3d_cuda.cu.so modulated_deform_im2col_3d_cuda.cu -x cu -Xcompiler -fPIC

# compile deform-psroi-pooling
nvcc -c -o deform_psroi_pooling_2d_cuda.cu.so deform_psroi_pooling_2d_cuda.cu -x cu -Xcompiler -fPIC
nvcc -c -o deform_psroi_pooling_3d_cuda.cu.so deform_psroi_pooling_3d_cuda.cu -x cu -Xcompiler -fPIC

cd ../..
CC=g++ python build_2d.py
python build_modulated_2d.py
CC=g++ python build_3d.py
python build_modulated_3d.py
