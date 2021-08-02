include_directories(${OptiX_INCLUDE})

add_executable(test
    optix7.h
    Renderer.h
    Renderer.cpp
    main.cpp
)

target_link_libraries(test
    ${optix_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
)
