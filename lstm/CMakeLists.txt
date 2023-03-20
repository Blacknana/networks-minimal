hunter_private_data(
    URL "https://lstm-1307949885.cos.ap-beijing.myqcloud.com/lstm_data.zip" 
    SHA1 "23fe399972006fbda0306ff5dceb7d93f1e0922b"
    FILE "lstm_data.zip"
    LOCATION lstm_data_path
)
file(ARCHIVE_EXTRACT INPUT ${lstm_data_path})

add_executable(wavefront_lstm lstm.cu)
target_link_libraries(wavefront_lstm gtest gtest_main CUDA::cuda_driver absl::base)

add_executable(naive_lstm base.cu)
target_link_libraries(naive_lstm gtest gtest_main CUDA::cuda_driver absl::base)