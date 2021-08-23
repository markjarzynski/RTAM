include_directories(${OPTIX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

cuda_add_executable(execute_test
    ${embedded_ptx_code}
    CUDABuffer.h
    LaunchParams.h
    Camera.h
    Triangle.h
    Renderer.h
    Renderer.cpp
    World.h
    World.cpp
    main.cpp
)

target_link_libraries(execute_test
    glad
    sutil_7_sdk
    ${OPTIX_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
)
