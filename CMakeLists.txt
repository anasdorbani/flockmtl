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

find_program(FLOCK_CARGO_EXECUTABLE cargo)
if(NOT FLOCK_CARGO_EXECUTABLE)
    message(FATAL_ERROR "cargo is required to build Flock because tokenizers-cpp is compiled into the extension")
endif()
if(NOT EXISTS "${FLOCK_DEFAULT_TOKENIZER_PATH}")
    message(FATAL_ERROR "Default tokenizer asset not found: ${FLOCK_DEFAULT_TOKENIZER_PATH}")
endif()

include(FetchContent)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG v0.1.1
    GIT_SHALLOW TRUE
    GIT_SUBMODULES_RECURSE TRUE)
FetchContent_GetProperties(tokenizers_cpp)
if(NOT tokenizers_cpp_POPULATED)
    FetchContent_Populate(tokenizers_cpp)
    set(TOKENIZERS_CPP_CARGO_TOML "${tokenizers_cpp_SOURCE_DIR}/rust/Cargo.toml")
    file(READ "${TOKENIZERS_CPP_CARGO_TOML}" TOKENIZERS_CPP_CARGO_TOML_CONTENTS)
    string(REPLACE
        "tokenizers = { version = \"0.21.0\","
        "tokenizers = { version = \"=0.21.0\","
        TOKENIZERS_CPP_CARGO_TOML_CONTENTS
        "${TOKENIZERS_CPP_CARGO_TOML_CONTENTS}")
    file(WRITE "${TOKENIZERS_CPP_CARGO_TOML}" "${TOKENIZERS_CPP_CARGO_TOML_CONTENTS}")
    file(REMOVE "${tokenizers_cpp_SOURCE_DIR}/rust/Cargo.lock")
    set(TOKENIZERS_CPP_RUST_LIB_RS "${tokenizers_cpp_SOURCE_DIR}/rust/src/lib.rs")
    file(READ "${TOKENIZERS_CPP_RUST_LIB_RS}" TOKENIZERS_CPP_RUST_LIB_RS_CONTENTS)
    string(REPLACE
        "(*handle).decode_str.len()"
        "(&(*handle).decode_str).len()"
        TOKENIZERS_CPP_RUST_LIB_RS_CONTENTS
        "${TOKENIZERS_CPP_RUST_LIB_RS_CONTENTS}")
    string(REPLACE
        "(*handle).id_to_token_result.len()"
        "(&(*handle).id_to_token_result).len()"
        TOKENIZERS_CPP_RUST_LIB_RS_CONTENTS
        "${TOKENIZERS_CPP_RUST_LIB_RS_CONTENTS}")
    file(WRITE "${TOKENIZERS_CPP_RUST_LIB_RS}" "${TOKENIZERS_CPP_RUST_LIB_RS_CONTENTS}")
    add_subdirectory(${tokenizers_cpp_SOURCE_DIR} ${tokenizers_cpp_BINARY_DIR})
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
target_link_libraries(${EXTENSION_NAME} tokenizers_cpp)
get_target_property(FLOCK_EXTENSION_INTERFACE_LIBS ${EXTENSION_NAME} INTERFACE_LINK_LIBRARIES)
if(FLOCK_EXTENSION_INTERFACE_LIBS)
    list(REMOVE_ITEM FLOCK_EXTENSION_INTERFACE_LIBS tokenizers_cpp "$<LINK_ONLY:tokenizers_cpp>")
    set_property(TARGET ${EXTENSION_NAME} PROPERTY INTERFACE_LINK_LIBRARIES "${FLOCK_EXTENSION_INTERFACE_LIBS}")
endif()

# Link libraries for the loadable extension
if(NOT EMSCRIPTEN)
    target_link_libraries(${LOADABLE_EXTENSION_NAME} CURL::libcurl)
endif()
target_link_libraries(${LOADABLE_EXTENSION_NAME} nlohmann_json::nlohmann_json)
target_compile_definitions(${LOADABLE_EXTENSION_NAME} PRIVATE
    FLOCK_DEFAULT_TOKENIZER_PATH="${FLOCK_DEFAULT_TOKENIZER_PATH}")
target_link_libraries(${LOADABLE_EXTENSION_NAME} tokenizers_cpp)

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
