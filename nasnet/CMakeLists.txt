hunter_private_data(
    URL "https://nasnet-1307949885.cos.ap-beijing.myqcloud.com/nasnet_data.zip" 
    SHA1 "7ea4e9e5f3d141d33eada6c3faafbb9b32256a1e"
    FILE "nasnet_data.zip"
    LOCATION nasnet_data_path
)
file(ARCHIVE_EXTRACT INPUT ${nasnet_data_path})

add_executable(nasnet nasnet.cpp)   
target_link_libraries(nasnet nasnet_kernel)

add_executable(nasnet_base nasnet.cpp)   
target_link_libraries(nasnet_base base_kernel)

add_executable(nasnet_rammer nasnet.cpp)
target_link_libraries(nasnet_rammer rammer_kernel)

add_subdirectory(kernels)
add_subdirectory(base_kernels)
add_subdirectory(rammer_kernels)
