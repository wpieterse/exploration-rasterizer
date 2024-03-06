add_library(
    exploration_rasterizer_library
)

target_sources(
    exploration_rasterizer_library
    PUBLIC
        library.hpp
        math.hpp
        linear_allocator.hpp
    PRIVATE
        library.cpp
        library_generic.cpp
        library_optimized.cpp
        library_avx2.cpp
        library_avx512.cpp
        library_cuda.cu
)

add_executable(
    exploration_rasterizer_executable
)

target_sources(
    exploration_rasterizer_executable
    PRIVATE
        main.cpp
        models/cow/vertices.hpp
        models/cow/vertex_indices.hpp
        models/cow/texture_coordinates.hpp
        models/cow/texture_coordinate_indices.hpp
)

target_link_libraries(
    exploration_rasterizer_executable
    PRIVATE
        exploration_rasterizer_library
)

set_target_properties(
    exploration_rasterizer_executable
    PROPERTIES
        OUTPUT_NAME main
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output/exploration/rasterizer
)

add_executable(
    exploration_rasterizer_bench
)

target_sources(
    exploration_rasterizer_bench
    PRIVATE
        bench.cpp
        models/cow/vertices.hpp
        models/cow/vertex_indices.hpp
        models/cow/texture_coordinates.hpp
        models/cow/texture_coordinate_indices.hpp
)

target_link_libraries(
    exploration_rasterizer_bench
    PRIVATE
        celero
        exploration_rasterizer_library
)

set_target_properties(
    exploration_rasterizer_bench
    PROPERTIES
        OUTPUT_NAME bench
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output/exploration/rasterizer
)
