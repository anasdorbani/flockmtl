#include "flock/registry/scalar.hpp"
#include "flock/functions/scalar/fusion_rrf.hpp"

namespace flock {


static void PragmaDisableProfiling(duckdb::ClientContext& context, duckdb::TableFunctionInput& data, duckdb::DataChunk& output) {

    // Put the result in the output
    std::cout << "Profiling disabled successfully" << std::endl;
}

static duckdb::unique_ptr<duckdb::FunctionData> BindSayHi(duckdb::ClientContext& context, duckdb::TableFunctionBindInput& input,
                                                          duckdb::vector<duckdb::LogicalType>& return_types, duckdb::vector<duckdb::string>& names) {
    // Set return types and column names
    return_types.emplace_back(duckdb::LogicalType::VARCHAR);
    names.emplace_back("message");

    // Process named parameters if needed
    for (auto& kv: input.named_parameters) {
        auto& name = kv.first;
        auto& value = kv.second;

        if (name == "level") {
            // Handle the level parameter
            duckdb::string level = value.ToString();
            std::cout << "Level: " << level << std::endl;
            // Do something with level...
        }
    }

    return nullptr;
}

void ScalarRegistry::Register(duckdb::ExtensionLoader& loader) {
    RegisterLlmComplete(loader);
    RegisterLlmEmbedding(loader);
    RegisterLlmFilter(loader);
    RegisterFusionRRF(loader);
    RegisterFusionCombANZ(loader);
    RegisterFusionCombMED(loader);
    RegisterFusionCombMNZ(loader);
    RegisterFusionCombSUM(loader);

    auto enable_fun = duckdb::TableFunction("say_hi", {}, PragmaDisableProfiling, BindSayHi, nullptr, nullptr);

    // Base config
    enable_fun.named_parameters.emplace("level", duckdb::LogicalType::VARCHAR);

    loader.RegisterFunction(enable_fun);
}

}// namespace flock
