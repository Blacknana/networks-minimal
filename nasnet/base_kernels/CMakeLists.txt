file(GLOB base_kernels *.cu)
add_library(base_kernel ${base_kernels})
target_link_libraries(base_kernel cudnn culibos cublas)
