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

if(NOT EXISTS "${FLOCK_DEFAULT_TOKENIZER_PATH}")
    message(FATAL_ERROR "Default tokenizer asset not found: ${FLOCK_DEFAULT_TOKENIZER_PATH}")
endif()

set(FLOCK_GENERATED_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/flock")
set(FLOCK_DEFAULT_TOKENIZER_SOURCE "${FLOCK_GENERATED_SOURCE_DIR}/embedded_default_tokenizer.cpp")
file(MAKE_DIRECTORY "${FLOCK_GENERATED_SOURCE_DIR}")
file(READ "${FLOCK_DEFAULT_TOKENIZER_PATH}" FLOCK_DEFAULT_TOKENIZER_HEX HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," FLOCK_DEFAULT_TOKENIZER_BYTES "${FLOCK_DEFAULT_TOKENIZER_HEX}")
string(REGEX REPLACE "((0x[0-9a-f][0-9a-f],){32})" "\\1\n" FLOCK_DEFAULT_TOKENIZER_BYTES "${FLOCK_DEFAULT_TOKENIZER_BYTES}")
file(WRITE "${FLOCK_DEFAULT_TOKENIZER_SOURCE}" "#include \"flock/functions/default_tokenizer_asset.hpp\"\n\n")
file(APPEND "${FLOCK_DEFAULT_TOKENIZER_SOURCE}" "namespace flock {\nnamespace {\n")
file(APPEND "${FLOCK_DEFAULT_TOKENIZER_SOURCE}" "alignas(16) constexpr unsigned char kDefaultTokenizerJson[] = {\n")
file(APPEND "${FLOCK_DEFAULT_TOKENIZER_SOURCE}" "${FLOCK_DEFAULT_TOKENIZER_BYTES}\n")
file(APPEND "${FLOCK_DEFAULT_TOKENIZER_SOURCE}" "};\n} // namespace\n\n")
file(APPEND "${FLOCK_DEFAULT_TOKENIZER_SOURCE}" "const char* DefaultTokenizerJsonData() {\n")
file(APPEND "${FLOCK_DEFAULT_TOKENIZER_SOURCE}" "    return reinterpret_cast<const char*>(kDefaultTokenizerJson);\n")
file(APPEND "${FLOCK_DEFAULT_TOKENIZER_SOURCE}" "}\n\n")
file(APPEND "${FLOCK_DEFAULT_TOKENIZER_SOURCE}" "size_t DefaultTokenizerJsonSize() {\n")
file(APPEND "${FLOCK_DEFAULT_TOKENIZER_SOURCE}" "    return sizeof(kDefaultTokenizerJson);\n")
file(APPEND "${FLOCK_DEFAULT_TOKENIZER_SOURCE}" "}\n")
file(APPEND "${FLOCK_DEFAULT_TOKENIZER_SOURCE}" "} // namespace flock\n")
set_source_files_properties("${FLOCK_DEFAULT_TOKENIZER_SOURCE}" PROPERTIES GENERATED TRUE)
list(APPEND EXTENSION_SOURCES "${FLOCK_DEFAULT_TOKENIZER_SOURCE}")

include(FetchContent)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

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

# Link libraries for the loadable extension
if(NOT EMSCRIPTEN)
    target_link_libraries(${LOADABLE_EXTENSION_NAME} CURL::libcurl)
endif()
target_link_libraries(${LOADABLE_EXTENSION_NAME} nlohmann_json::nlohmann_json)

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
