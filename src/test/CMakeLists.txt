include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

add_executable(test
    ${embedded_ptx_code}
    CUDABuffer.h
    Renderer.h
    Renderer.cpp
    World.h
    World.cpp
    main.cpp
)

target_link_libraries(test
    glad
    ${optix_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
)
