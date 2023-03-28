add_executable(deepspeech2 deepspeech.cu)
target_link_libraries(deepspeech2 PUBLIC pthread cudart gtest gtest_main deepspeech_kernels  deepspeech_lstm)

add_executable(deepspeech2_base deepspeech.cu)
target_link_libraries(deepspeech2_base PUBLIC pthread cudart gtest gtest_main deepspeech_kernels  deepspeech_lstm_base)

hunter_private_data(
    URL "https://lstm-1307949885.cos.ap-beijing.myqcloud.com/deepspeech2_data.zip" 
    SHA1 "c0b8bc7ad5532b311ac7ee71bcc70cb7005dde17"
    FILE "deepspeech2_data.zip"
    LOCATION deepspeech2_data_path
)
file(ARCHIVE_EXTRACT INPUT ${deepspeech2_data_path})

add_subdirectory(op)
add_subdirectory(lstm)