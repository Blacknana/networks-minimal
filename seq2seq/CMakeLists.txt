hunter_private_data(
    URL "https://lstm-1307949885.cos.ap-beijing.myqcloud.com/seq2seq_data.zip" 
    SHA1 "db437a915e28c3a41d7d958fcbd3f1b144b2d00f"
    FILE "seq2seq_data.zip"
    LOCATION seq2seq_data_path
)
file(ARCHIVE_EXTRACT INPUT ${seq2seq_data_path})

add_executable(seq2seq seq2seq.cu)
target_link_libraries(seq2seq gtest gtest_main CUDA::cuda_driver absl::base)

add_executable(seq2seq_base base.cu)
target_link_libraries(seq2seq_base gtest gtest_main CUDA::cuda_driver absl::base)