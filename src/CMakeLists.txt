set(RENDERER_SIMULATOR_SRCS
    types.hpp
    sim.hpp sim.inl sim.cpp
)
SET(RENDERER_COMPILE_FLAGS
    -v)

add_library(renderer_cpu_impl STATIC
    ${RENDERER_SIMULATOR_SRCS}
)

target_link_libraries(renderer_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
        madrona_rendering_system
)

add_library(renderer_mgr STATIC
    mgr.hpp mgr.cpp
)

add_library(run_common STATIC
    args.hpp args.cpp
    dump.hpp dump.cpp
)

target_include_directories(run_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(run_common
    PRIVATE
        madrona_libcxx
        madrona_cuda
        madrona_mw_core
        stb
)

target_link_libraries(renderer_mgr 
    PUBLIC
        madrona_python_utils
    PRIVATE
        renderer_cpu_impl
        madrona_mw_cpu
        madrona_common
        madrona_importer
        madrona_physics_loader
        madrona_render
        madrona_ktx
        madrona_render_asset_processor
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            renderer_gpu_srcs
        SOURCES_DEFN
            RENDERER_SRC_LIST
        FLAGS_DEFN
            RENDERER_COMPILE_FLAGS 
        SRCS
            ${RENDERER_SIMULATOR_SRCS}
    )

    target_link_libraries(renderer_mgr PRIVATE
        renderer_gpu_srcs
        madrona_mw_gpu
    )
endif ()

target_compile_definitions(renderer_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

if (TARGET madrona_viz)
    add_executable(renderer_viewer viewer.cpp)
    target_link_libraries(renderer_viewer 
        PUBLIC
        renderer_mgr
        PRIVATE
        madrona_mw_core madrona_viz madrona_cuda stb run_common)

    target_compile_definitions(renderer_viewer PUBLIC
        -DMADRONA_VIEWER
    )

    if (TARGET madrona_cuda)
        target_link_libraries(renderer_viewer PRIVATE madrona_cuda)
    endif()
    target_compile_definitions(renderer_viewer PRIVATE
        -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
    )
endif()

add_executable(renderer_headless headless.cpp)
target_link_libraries(renderer_headless 
    PUBLIC 
        madrona_mw_core renderer_mgr madrona_viz madrona_cuda stb run_common
)
