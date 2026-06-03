#pragma once

#include "flock/model_manager/repository.hpp"
#include <functional>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace flock {

class PromptTokenizer {
public:
    using TokenCounter = std::function<size_t(const std::string&)>;

    static size_t CountTokens(const std::string& text);
    static void SetTokenCounterForTesting(TokenCounter counter);
};

class PromptBatcher {
public:
    using PromptBuilder = std::function<std::string(int tuple_count)>;

    static int UsableContext(const ModelDetails& model_details);
    static int EffectiveBatchSize(const ModelDetails& model_details, int remaining_tuples);
    static int FindMaxTupleCount(const ModelDetails& model_details,
                                 int max_tuple_count,
                                 const PromptBuilder& prompt_builder,
                                 const std::string& function_name);

    static nlohmann::json SliceColumns(const nlohmann::json& columns, int start_index, int count);
    static void AppendRows(nlohmann::json& target, const nlohmann::json& source, int start_index, int count);
    static nlohmann::json AddLocalRowIds(const nlohmann::json& columns);
    static int RowCount(const nlohmann::json& columns);

    static std::vector<std::vector<std::string>> BatchEmbeddingInputs(const std::vector<std::string>& inputs,
                                                                      const ModelDetails& model_details);
};

}// namespace flock
