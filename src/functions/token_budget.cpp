#include "flock/functions/token_budget.hpp"

#include "fmt/format.h"
#include <algorithm>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include <tokenizers_cpp.h>

namespace flock {

namespace {

std::mutex& CounterMutex() {
    static std::mutex mutex;
    return mutex;
}

PromptTokenizer::TokenCounter& TestCounter() {
    static PromptTokenizer::TokenCounter counter;
    return counter;
}

std::mutex& TokenizerMutex() {
    static std::mutex mutex;
    return mutex;
}

std::string LoadFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        return "";
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

tokenizers::Tokenizer& DefaultTokenizer() {
    static std::unique_ptr<tokenizers::Tokenizer> tokenizer = []() {
#ifdef FLOCK_DEFAULT_TOKENIZER_PATH
        auto blob = LoadFile(FLOCK_DEFAULT_TOKENIZER_PATH);
        if (!blob.empty()) {
            auto loaded_tokenizer = tokenizers::Tokenizer::FromBlobJSON(blob);
            if (loaded_tokenizer) {
                return loaded_tokenizer;
            }
        }
#endif
        throw std::runtime_error(
                "Failed to load the bundled default tokenizer from FLOCK_DEFAULT_TOKENIZER_PATH");
    }();
    return *tokenizer;
}

std::string BudgetExceededMessage(const std::string& function_name,
                                  size_t prompt_tokens,
                                  int usable_context,
                                  const ModelDetails& model_details) {
    return duckdb_fmt::format(
            "{} prompt token budget exceeded: a single tuple requires {} tokens, but the usable context is {} tokens "
            "(context_window={}, safe_margin={}). Reduce the row/input size, increase context_window, or lower safe_margin.",
            function_name, prompt_tokens, usable_context, model_details.context_window, model_details.safe_margin);
}

}// namespace

size_t PromptTokenizer::CountTokens(const std::string& text) {
    {
        std::lock_guard<std::mutex> lock(CounterMutex());
        if (TestCounter()) {
            return TestCounter()(text);
        }
    }

    std::lock_guard<std::mutex> lock(TokenizerMutex());
    return DefaultTokenizer().Encode(text).size();
}

void PromptTokenizer::InitializeDefaultTokenizer() {
    std::lock_guard<std::mutex> lock(TokenizerMutex());
    (void)DefaultTokenizer();
}

void PromptTokenizer::SetTokenCounterForTesting(TokenCounter counter) {
    std::lock_guard<std::mutex> lock(CounterMutex());
    TestCounter() = std::move(counter);
}

int PromptBatcher::UsableContext(const ModelDetails& model_details) {
    if (model_details.context_window <= 0) {
        throw std::runtime_error("context_window must be greater than zero");
    }
    if (model_details.safe_margin < 0) {
        throw std::runtime_error("safe_margin must be greater than or equal to zero");
    }
    if (model_details.context_window <= model_details.safe_margin) {
        throw std::runtime_error("context_window must be greater than safe_margin");
    }
    return model_details.context_window - model_details.safe_margin;
}

int PromptBatcher::EffectiveBatchSize(const ModelDetails& model_details, int remaining_tuples) {
    if (model_details.batch_size <= 0) {
        throw std::runtime_error("Batch size must be greater than zero");
    }
    return std::min(model_details.batch_size, remaining_tuples);
}

int PromptBatcher::FindMaxTupleCount(const ModelDetails& model_details,
                                     int max_tuple_count,
                                     const PromptBuilder& prompt_builder,
                                     const std::string& function_name) {
    if (max_tuple_count <= 0) {
        return 0;
    }

    const auto usable_context = UsableContext(model_details);
    int best_count = 0;
    size_t first_count_tokens = 0;

    for (int count = 1; count <= max_tuple_count; count++) {
        const auto prompt = prompt_builder(count);
        const auto prompt_tokens = PromptTokenizer::CountTokens(prompt);
        if (count == 1) {
            first_count_tokens = prompt_tokens;
        }
        if (static_cast<int64_t>(prompt_tokens) > usable_context) {
            break;
        }
        best_count = count;
    }

    if (best_count == 0) {
        throw std::runtime_error(BudgetExceededMessage(function_name, first_count_tokens, usable_context, model_details));
    }
    return best_count;
}

nlohmann::json PromptBatcher::SliceColumns(const nlohmann::json& columns, int start_index, int count) {
    nlohmann::json result = nlohmann::json::array();
    for (const auto& column: columns) {
        nlohmann::json output_column = nlohmann::json::object();
        for (const auto& item: column.items()) {
            if (item.key() == "data") {
                output_column["data"] = nlohmann::json::array();
                for (int row_idx = 0; row_idx < count && start_index + row_idx < static_cast<int>(item.value().size()); row_idx++) {
                    output_column["data"].push_back(item.value()[start_index + row_idx]);
                }
            } else {
                output_column[item.key()] = item.value();
            }
        }
        result.push_back(output_column);
    }
    return result;
}

void PromptBatcher::AppendRows(nlohmann::json& target, const nlohmann::json& source, int start_index, int count) {
    if (target.empty()) {
        for (const auto& column: source) {
            nlohmann::json output_column = nlohmann::json::object();
            for (const auto& item: column.items()) {
                if (item.key() == "data") {
                    output_column["data"] = nlohmann::json::array();
                } else {
                    output_column[item.key()] = item.value();
                }
            }
            target.push_back(output_column);
        }
    }

    for (auto column_idx = 0; column_idx < static_cast<int>(source.size()); column_idx++) {
        if (!target[column_idx].contains("data")) {
            target[column_idx]["data"] = nlohmann::json::array();
        }
        const auto& source_data = source[column_idx]["data"];
        for (int row_idx = 0; row_idx < count && start_index + row_idx < static_cast<int>(source_data.size()); row_idx++) {
            target[column_idx]["data"].push_back(source_data[start_index + row_idx]);
        }
    }
}

nlohmann::json PromptBatcher::AddLocalRowIds(const nlohmann::json& columns) {
    auto indexed_columns = columns;
    nlohmann::json row_id_column = {{"name", "flock_row_id"}, {"data", nlohmann::json::array()}};
    for (int row_idx = 0; row_idx < RowCount(columns); row_idx++) {
        row_id_column["data"].push_back(std::to_string(row_idx));
    }
    indexed_columns.push_back(row_id_column);
    return indexed_columns;
}

int PromptBatcher::RowCount(const nlohmann::json& columns) {
    if (columns.empty() || !columns[0].contains("data")) {
        return 0;
    }
    return static_cast<int>(columns[0]["data"].size());
}

std::vector<std::vector<std::string>> PromptBatcher::BatchEmbeddingInputs(const std::vector<std::string>& inputs,
                                                                          const ModelDetails& model_details) {
    std::vector<std::vector<std::string>> batches;
    const auto usable_context = UsableContext(model_details);
    const auto configured_batch_size = model_details.batch_size;
    if (configured_batch_size <= 0) {
        throw std::runtime_error("Batch size must be greater than zero");
    }

    size_t index = 0;
    while (index < inputs.size()) {
        std::vector<std::string> batch;
        size_t token_total = 0;
        while (index < inputs.size() && static_cast<int>(batch.size()) < configured_batch_size) {
            const auto input_tokens = PromptTokenizer::CountTokens(inputs[index]);
            if (static_cast<int64_t>(input_tokens) > usable_context) {
                throw std::runtime_error(BudgetExceededMessage("llm_embedding", input_tokens, usable_context, model_details));
            }
            if (!batch.empty() && static_cast<int64_t>(token_total + input_tokens) > usable_context) {
                break;
            }
            token_total += input_tokens;
            batch.push_back(inputs[index]);
            index++;
        }
        batches.push_back(std::move(batch));
    }

    return batches;
}

}// namespace flock
