cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set extension name here
set(TARGET_NAME flock)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

# Include directories
include_directories(src/include)
add_subdirectory(src)

# Find dependencies
if(NOT EMSCRIPTEN)
    find_package(CURL REQUIRED)
endif()
find_package(nlohmann_json CONFIG REQUIRED)

set(FLOCK_DEFAULT_TOKENIZER_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/resources/tokenizers/cl100k_base/tokenizer.json"
    CACHE STRING "Path to the bundled default tokenizer.json")

option(FLOCK_USE_TOKENIZERS_CPP "Use mlc-ai/tokenizers-cpp for prompt token counting" ON)
find_program(FLOCK_CARGO_EXECUTABLE cargo)
if(FLOCK_USE_TOKENIZERS_CPP AND FLOCK_CARGO_EXECUTABLE)
    include(FetchContent)
    set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        tokenizers_cpp
        GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
        GIT_TAG v0.1.1
        GIT_SHALLOW TRUE
        GIT_SUBMODULES_RECURSE TRUE)
    FetchContent_MakeAvailable(tokenizers_cpp)
    set(FLOCK_TOKENIZERS_CPP_ENABLED TRUE)
elseif(FLOCK_USE_TOKENIZERS_CPP)
    message(WARNING "cargo was not found; building Flock with the lightweight fallback token counter")
endif()

# Coverage instrumentation (must be before targets are built)
if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
    message(STATUS "Enabling code coverage")
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate -fcoverage-mapping)
endif()

# Build the DuckDB static and loadable extensions
build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Link the filesystem library for GCC < 9
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION
                                            VERSION_LESS 9.0)
  target_link_libraries(${EXTENSION_NAME} -lstdc++fs)
endif()

# Link libraries for the static extension
if(NOT EMSCRIPTEN)
    target_link_libraries(${EXTENSION_NAME} CURL::libcurl)
endif()
target_link_libraries(${EXTENSION_NAME} nlohmann_json::nlohmann_json)
target_compile_definitions(${EXTENSION_NAME} PRIVATE
    FLOCK_DEFAULT_TOKENIZER_PATH="${FLOCK_DEFAULT_TOKENIZER_PATH}")
if(FLOCK_TOKENIZERS_CPP_ENABLED)
    target_compile_definitions(${EXTENSION_NAME} PRIVATE FLOCK_ENABLE_TOKENIZERS_CPP)
    target_link_libraries(${EXTENSION_NAME} tokenizers_cpp)
endif()

# Link libraries for the loadable extension
if(NOT EMSCRIPTEN)
    target_link_libraries(${LOADABLE_EXTENSION_NAME} CURL::libcurl)
endif()
target_link_libraries(${LOADABLE_EXTENSION_NAME} nlohmann_json::nlohmann_json)
target_compile_definitions(${LOADABLE_EXTENSION_NAME} PRIVATE
    FLOCK_DEFAULT_TOKENIZER_PATH="${FLOCK_DEFAULT_TOKENIZER_PATH}")
if(FLOCK_TOKENIZERS_CPP_ENABLED)
    target_compile_definitions(${LOADABLE_EXTENSION_NAME} PRIVATE FLOCK_ENABLE_TOKENIZERS_CPP)
    target_link_libraries(${LOADABLE_EXTENSION_NAME} tokenizers_cpp)
endif()

# Install the extension
install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

if(NOT EMSCRIPTEN)
    # Add the test directory if not on WASM
    enable_testing()
    add_subdirectory(test/unit)
endif()
