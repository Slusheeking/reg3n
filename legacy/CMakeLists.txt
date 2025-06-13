cmake_minimum_required(VERSION 3.20)
project(UltraFastHFT VERSION 1.0.0 LANGUAGES CXX)

# Set C++17 standard for Beast compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Production optimization options
option(ENABLE_NATIVE_ARCH "Enable native architecture optimizations" ON)
option(ENABLE_PGO "Enable Profile-Guided Optimization" OFF)
option(ENABLE_LTO "Enable Link-Time Optimization" ON)
option(ENABLE_AVX512 "Enable AVX-512 instructions if available" OFF)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Base optimization flags
    if(ENABLE_NATIVE_ARCH)
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=x86-64-v3 -mtune=generic")
    endif()
    
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address -fsanitize=undefined")
    
    # Link-Time Optimization
    if(ENABLE_LTO)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto=auto")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto=auto")
    endif()
    
    # Boost compatibility for Beast HTTP client
    add_definitions(-DBOOST_BIND_GLOBAL_PLACEHOLDERS)
    
    # Ultra-low latency optimizations
    add_compile_options(
        -ffast-math
        -funroll-loops
        -finline-functions
        -fomit-frame-pointer
        -falign-functions=32
        -falign-loops=32
        -falign-jumps=32
        -msse4.2
        -mavx
        -mavx2
        -mfma
        -mbmi
        -mbmi2
        -mlzcnt
        -mpopcnt
        -mf16c
        -fprefetch-loop-arrays
        -ftracer
        -fweb
        -frename-registers
        -fgcse-after-reload
        -ftree-vectorize
        -fvect-cost-model=unlimited
    )
    
    # Check for AVX-512 support
    include(CheckCXXCompilerFlag)
    if(ENABLE_AVX512)
        check_cxx_compiler_flag("-mavx512f" COMPILER_HAS_AVX512F)
        check_cxx_compiler_flag("-mavx512dq" COMPILER_HAS_AVX512DQ)
        check_cxx_compiler_flag("-mavx512bw" COMPILER_HAS_AVX512BW)
        check_cxx_compiler_flag("-mavx512vl" COMPILER_HAS_AVX512VL)
        
        if(COMPILER_HAS_AVX512F AND COMPILER_HAS_AVX512DQ AND
           COMPILER_HAS_AVX512BW AND COMPILER_HAS_AVX512VL)
            set(COMPILER_SUPPORTS_AVX512 TRUE)
            add_compile_options(-mavx512f -mavx512dq -mavx512bw -mavx512vl)
            message(STATUS "AVX-512 support enabled")
        else()
            set(COMPILER_SUPPORTS_AVX512 FALSE)
            message(STATUS "AVX-512 not supported by compiler")
        endif()
    else()
        set(COMPILER_SUPPORTS_AVX512 FALSE)
        message(STATUS "AVX-512 disabled by user")
    endif()
    
    # Profile-Guided Optimization setup
    if(ENABLE_PGO)
        if(NOT DEFINED PGO_PHASE)
            set(PGO_PHASE "generate" CACHE STRING "PGO phase: generate or use")
        endif()
        
        if(PGO_PHASE STREQUAL "generate")
            add_compile_options(-fprofile-generate=${CMAKE_BINARY_DIR}/pgo-data)
            add_link_options(-fprofile-generate=${CMAKE_BINARY_DIR}/pgo-data)
            message(STATUS "PGO: Profile generation enabled")
        elseif(PGO_PHASE STREQUAL "use")
            if(EXISTS ${CMAKE_BINARY_DIR}/pgo-data)
                add_compile_options(-fprofile-use=${CMAKE_BINARY_DIR}/pgo-data -fprofile-correction)
                add_link_options(-fprofile-use=${CMAKE_BINARY_DIR}/pgo-data)
                message(STATUS "PGO: Profile optimization enabled")
            else()
                message(WARNING "PGO: Profile data not found at ${CMAKE_BINARY_DIR}/pgo-data")
            endif()
        endif()
    endif()
    
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -xHost -ipo -no-prec-div -fp-model fast=2")
    add_compile_options(-qopt-report=5)
endif()

# Find required packages
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

# Find Boost (header-only for Beast)
find_package(Boost 1.70.0 REQUIRED)

# Find WebSocket++ (header-only library)
find_path(WEBSOCKETPP_INCLUDE_DIR websocketpp/config/asio_client.hpp
    PATHS /usr/include /usr/local/include
)

if(WEBSOCKETPP_INCLUDE_DIR)
    message(STATUS "WebSocket++ found at ${WEBSOCKETPP_INCLUDE_DIR}")
else()
    message(WARNING "WebSocket++ not found - install libwebsocketpp-dev")
endif()

if(Boost_FOUND)
    message(STATUS "Boost found: ${Boost_VERSION}")
    message(STATUS "Boost include: ${Boost_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Boost not found - install libboost-all-dev")
endif()

# Find RapidJSON (required for JSON parsing)
find_path(RAPIDJSON_INCLUDE_DIR rapidjson/rapidjson.h
    PATHS /usr/include /usr/local/include
)


# Check for simdjson
find_path(SIMDJSON_INCLUDE_DIR simdjson.h
    PATHS /usr/include /usr/local/include
)
find_library(SIMDJSON_LIBRARY simdjson
    PATHS /usr/lib /usr/local/lib
)

if(RAPIDJSON_INCLUDE_DIR)
    message(STATUS "RapidJSON found at ${RAPIDJSON_INCLUDE_DIR}")
else()
    message(WARNING "RapidJSON not found - install rapidjson-dev")
endif()


if(SIMDJSON_INCLUDE_DIR AND SIMDJSON_LIBRARY)
    message(STATUS "simdjson found")
else()
    message(WARNING "simdjson not found - install libsimdjson-dev")
endif()

# Check for NUMA support
find_path(NUMA_INCLUDE_DIR numa.h)
find_library(NUMA_LIBRARY numa)

if(NUMA_INCLUDE_DIR AND NUMA_LIBRARY)
    set(NUMA_FOUND TRUE)
    message(STATUS "NUMA support found")
else()
    set(NUMA_FOUND FALSE)
    message(WARNING "NUMA support not found - install libnuma-dev for optimal performance")
endif()

# Check for Intel TBB
find_package(PkgConfig REQUIRED)
pkg_check_modules(TBB tbb)
if(TBB_FOUND)
    message(STATUS "Intel TBB found")
else()
    message(WARNING "Intel TBB not found - some optimizations may be unavailable")
endif()

# Check for Intel MKL
set(MKL_ROOT "/usr")
find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKL_ROOT}/include)
find_library(MKL_RT_LIBRARY mkl_rt PATHS ${MKL_ROOT}/lib/x86_64-linux-gnu)

if(NOT MKL_RT_LIBRARY)
    find_library(MKL_RT_LIBRARY libmkl_rt.so PATHS ${MKL_ROOT}/lib/x86_64-linux-gnu)
endif()

if(MKL_INCLUDE_DIR AND MKL_RT_LIBRARY)
    set(MKL_FOUND TRUE)
    message(STATUS "Intel MKL found at ${MKL_RT_LIBRARY}")
else()
    pkg_check_modules(MKL mkl-dynamic-lp64-seq)
    if(MKL_FOUND)
        message(STATUS "Intel MKL found via pkg-config")
    else()
        message(WARNING "Intel MKL not found - using standard math libraries")
    endif()
endif()

# Create the main executable with WebSocket + REST API implementation
add_executable(hft_engine
    main.cpp
    beast_http_client.cpp
    alpaca_client.cpp
    websocket_client.cpp
)

# Set target properties - KEEP C++17 to match global setting
set_target_properties(hft_engine PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    INTERPROCEDURAL_OPTIMIZATION FALSE
)

# Include directories
target_include_directories(hft_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
    /usr/include/rapidjson
)

# Add conditional includes
if(RAPIDJSON_INCLUDE_DIR)
    target_include_directories(hft_engine PRIVATE ${RAPIDJSON_INCLUDE_DIR})
    target_compile_definitions(hft_engine PRIVATE HAVE_RAPIDJSON)
endif()

if(WEBSOCKETPP_INCLUDE_DIR)
    target_include_directories(hft_engine PRIVATE ${WEBSOCKETPP_INCLUDE_DIR})
    target_compile_definitions(hft_engine PRIVATE HAVE_WEBSOCKETPP)
endif()


if(SIMDJSON_INCLUDE_DIR)
    target_include_directories(hft_engine PRIVATE ${SIMDJSON_INCLUDE_DIR})
    target_compile_definitions(hft_engine PRIVATE HAVE_SIMDJSON)
endif()

# Link libraries - Boost.Beast is header-only
target_link_libraries(hft_engine PRIVATE
    Threads::Threads
    pthread
    rt
    OpenSSL::SSL
    OpenSSL::Crypto
    z
)

if(SIMDJSON_LIBRARY)
    target_link_libraries(hft_engine PRIVATE ${SIMDJSON_LIBRARY})
endif()

if(NUMA_FOUND)
    target_include_directories(hft_engine PRIVATE ${NUMA_INCLUDE_DIR})
    target_link_libraries(hft_engine PRIVATE ${NUMA_LIBRARY})
    target_compile_definitions(hft_engine PRIVATE HAVE_NUMA)
endif()

if(TBB_FOUND)
    target_include_directories(hft_engine PRIVATE ${TBB_INCLUDE_DIRS})
    target_link_libraries(hft_engine PRIVATE ${TBB_LIBRARIES})
    target_compile_definitions(hft_engine PRIVATE HAVE_TBB)
endif()

if(MKL_FOUND)
    if(MKL_INCLUDE_DIR)
        target_include_directories(hft_engine PRIVATE ${MKL_INCLUDE_DIR})
    elseif(MKL_INCLUDE_DIRS)
        target_include_directories(hft_engine PRIVATE ${MKL_INCLUDE_DIRS})
    endif()
    
    if(MKL_RT_LIBRARY)
        target_link_libraries(hft_engine PRIVATE ${MKL_RT_LIBRARY})
    elseif(MKL_LIBRARIES)
        target_link_libraries(hft_engine PRIVATE ${MKL_LIBRARIES})
    endif()
    
    target_compile_definitions(hft_engine PRIVATE HAVE_MKL)
endif()

# Add Beast-specific compile definitions
target_compile_definitions(hft_engine PRIVATE
    BOOST_BEAST_USE_STD_STRING_VIEW
    BOOST_ASIO_HAS_STD_CHRONO
    BOOST_ASIO_HEADER_ONLY=1
    HAVE_BOOST_BEAST
)

# Platform-specific optimizations
if(UNIX AND NOT APPLE)
    target_compile_definitions(hft_engine PRIVATE _GNU_SOURCE)
    target_link_libraries(hft_engine PRIVATE dl)
    target_compile_definitions(hft_engine PRIVATE HAVE_HUGE_PAGES)
    target_compile_definitions(hft_engine PRIVATE HAVE_SCHED_FIFO)
endif()

# Debug configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(hft_engine PRIVATE DEBUG_BUILD)
    target_compile_options(hft_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Release configuration
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(hft_engine PRIVATE RELEASE_BUILD NDEBUG)
    
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        option(ENABLE_PGO "Enable Profile-Guided Optimization" OFF)
        if(ENABLE_PGO)
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-generate")
            message(STATUS "PGO training build enabled - run the application to generate profile data")
        endif()
    endif()
endif()

# Custom targets for performance testing and PGO
add_custom_target(test_performance
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/hft_engine --test
    DEPENDS hft_engine
    COMMENT "Running HFT performance test"
)

add_custom_target(benchmark_latency
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/hft_engine --test --numa-node 0
    DEPENDS hft_engine
    COMMENT "Running latency benchmark with NUMA optimization"
)

# Profile-Guided Optimization workflow
add_custom_target(pgo_generate
    COMMAND ${CMAKE_COMMAND} -DENABLE_PGO=ON -DPGO_PHASE=generate -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target hft_engine
    COMMENT "Building for PGO profile generation"
)

add_custom_target(pgo_train
    COMMAND mkdir -p ${CMAKE_BINARY_DIR}/pgo-data
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/hft_engine --test --duration 60
    DEPENDS pgo_generate
    COMMENT "Training PGO with representative workload"
)

add_custom_target(pgo_optimize
    COMMAND ${CMAKE_COMMAND} -DENABLE_PGO=ON -DPGO_PHASE=use -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target hft_engine
    DEPENDS pgo_train
    COMMENT "Building optimized binary with PGO"
)

# Complete PGO workflow
add_custom_target(pgo_full
    DEPENDS pgo_optimize
    COMMENT "Complete PGO workflow: generate -> train -> optimize"
)

# Comprehensive benchmark suite
add_custom_target(benchmark_comprehensive
    COMMAND echo "=== Comprehensive HFT Benchmark Suite ==="
    COMMAND echo "Pipeline latency test:"
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/hft_engine --test --benchmark pipeline_latency
    COMMAND echo "Memory allocation test:"
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/hft_engine --test --benchmark memory_allocation
    COMMAND echo "Model inference test:"
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/hft_engine --test --benchmark model_inference
    COMMAND echo "REST API throughput test:"
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/hft_engine --test --benchmark rest_api_throughput
    DEPENDS hft_engine
    COMMENT "Running comprehensive benchmark suite"
)

# Production deployment preparation
add_custom_target(prepare_production
    COMMAND echo "Preparing production deployment..."
    COMMAND chmod +x ${CMAKE_SOURCE_DIR}/optimize_system.sh
    COMMAND chmod +x ${CMAKE_SOURCE_DIR}/run_hft_optimized.sh || true
    COMMAND cp ${CMAKE_SOURCE_DIR}/hft-optimize.service ${CMAKE_BINARY_DIR}/
    COMMAND echo "Production files ready in build directory"
    DEPENDS hft_engine
    COMMENT "Preparing production deployment files"
)

# Installation
install(TARGETS hft_engine
    RUNTIME DESTINATION bin
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Ultra-Fast HFT Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Boost Version: ${Boost_VERSION}")
message(STATUS "Beast HTTP Client: Enabled")
message(STATUS "WebSocket: Enabled (Polygon.io real-time data)")
message(STATUS "Rate Limit: 200 req/min (30s polling)")
message(STATUS "NUMA Support: ${NUMA_FOUND}")
message(STATUS "Intel TBB: ${TBB_FOUND}")
message(STATUS "Intel MKL: ${MKL_FOUND}")
message(STATUS "AVX-512: ${COMPILER_SUPPORTS_AVX512}")
message(STATUS "Target: Sub-200μs pipeline latency")
message(STATUS "==========================================")
message(STATUS "")
