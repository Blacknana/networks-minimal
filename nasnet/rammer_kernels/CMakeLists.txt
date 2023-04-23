file(GLOB rammer_kernels *.cu) 
add_library(rammer_kernel ${rammer_kernels})
target_link_libraries(rammer_kernel cudnn culibos cublas)
