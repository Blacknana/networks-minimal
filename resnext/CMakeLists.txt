add_executable(resnext resnext.cu)
target_link_libraries(resnext PUBLIC pthread cudart gtest gtest_main resnext_kernels)

hunter_private_data(
    URL "https://resnext-1307949885.cos.ap-nanjing.myqcloud.com/resnext_data.zip" 
    SHA1 "2ec68452f309d4d9028c1aad07a4e19a32a81410"
    FILE "resnext_data.zip"
    LOCATION resnext_data_path
)
file(ARCHIVE_EXTRACT INPUT ${resnext_data_path})

add_subdirectory(op)