hunter_private_data(
    URL "https://bert-1307949885.cos.ap-beijing.myqcloud.com/bert_data.zip" 
    SHA1 "ea5354a794650aa62fbd91e1fe8374b23fb07736"
    FILE "bert_data.zip"
    LOCATION bert_data_path
)
file(ARCHIVE_EXTRACT INPUT ${bert_data_path})

add_executable(bert bert.cu)
target_link_libraries(bert gtest gtest_main CUDA::cuda_driver absl::base)

add_executable(bert_base base.cu)
target_link_libraries(bert_base gtest gtest_main CUDA::cuda_driver absl::base)