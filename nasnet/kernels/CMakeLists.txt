file(GLOB nasnet_kernels *.cu) 
add_library(nasnet_kernel ${nasnet_kernels})
target_link_libraries(nasnet_kernel cudnn culibos cublas)
