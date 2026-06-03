#include "flock/functions/default_tokenizer_asset.hpp"
#include "flock/functions/token_budget.hpp"

#include "fmt/format.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <cctype>

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

enum class SegmentKind {
    AsciiWhitespace,
    AsciiWord,
    AsciiDigit,
    AsciiPunctuation,
    AsciiSymbol,
    Unicode,
    Control
};

struct TokenizerProfile {
    size_t vocab_size = 0;
    size_t merge_count = 0;
    size_t special_token_count = 0;
    std::vector<std::string> special_tokens;

    bool Loaded() const {
        return !special_tokens.empty() || vocab_size > 0;
    }
};

size_t CountUtf8Bytes(const std::string& text, size_t start_index) {
    const auto current = static_cast<uint8_t>(text[start_index]);
    if ((current & 0x80) == 0) {
        return 1;
    }
    if ((current & 0xE0) == 0xC0) {
        return (start_index + 1 < text.size()) ? 2 : 1;
    }
    if ((current & 0xF0) == 0xE0) {
        return (start_index + 2 < text.size()) ? 3 : 1;
    }
    if ((current & 0xF8) == 0xF0) {
        return (start_index + 3 < text.size()) ? 4 : 1;
    }
    return 1;
}

SegmentKind ClassifyAscii(uint8_t byte) {
    if (std::isspace(byte)) {
        return SegmentKind::AsciiWhitespace;
    }
    if (std::isdigit(byte)) {
        return SegmentKind::AsciiDigit;
    }
    if (std::isalpha(byte) || byte == '_' || byte == '\'') {
        return SegmentKind::AsciiWord;
    }
    if (std::isprint(byte)) {
        return std::ispunct(byte) ? SegmentKind::AsciiPunctuation : SegmentKind::AsciiSymbol;
    }
    return SegmentKind::Control;
}

TokenizerProfile LoadTokenizerProfile() {
    TokenizerProfile profile;
    const std::string tokenizer_json(DefaultTokenizerJsonData(), DefaultTokenizerJsonSize());
    if (tokenizer_json.empty()) {
        throw std::runtime_error("Failed to load the embedded default tokenizer asset.");
    }

    const auto json = nlohmann::json::parse(tokenizer_json);
    if (!json.is_object()) {
        throw std::runtime_error("Default tokenizer asset is not a valid JSON object.");
    }

    const auto model = json.value("model", nlohmann::json::object());
    if (model.contains("vocab") && model["vocab"].is_object()) {
        profile.vocab_size = model["vocab"].size();
    }
    if (model.contains("merges") && model["merges"].is_array()) {
        profile.merge_count = model["merges"].size();
    }
    if (json.contains("added_tokens") && json["added_tokens"].is_array()) {
        for (const auto& token : json["added_tokens"]) {
            if (!token.is_object()) {
                continue;
            }
            if (token.value("special", false)) {
                const auto content = token.value("content", "");
                if (content.empty()) {
                    continue;
                }
                profile.special_tokens.push_back(content);
            }
        }
    }
    profile.special_token_count = profile.special_tokens.size();

    if (profile.vocab_size == 0 && profile.merge_count == 0) {
        throw std::runtime_error("Default tokenizer profile is missing expected metadata.");
    }

    return profile;
}

const TokenizerProfile& DefaultTokenizerProfile() {
    static TokenizerProfile profile = []() {
        try {
            return LoadTokenizerProfile();
        } catch (...) {
            TokenizerProfile fallback;
            fallback.vocab_size = 100000;
            fallback.merge_count = 0;
            fallback.special_tokens = {"<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|endofprompt|>"};
            fallback.special_token_count = fallback.special_tokens.size();
            return fallback;
        }
    }();
    return profile;
}

size_t CountSpecialTokenOverhead(std::string_view text, const TokenizerProfile& profile) {
    size_t overhead = 0;
    for (const auto& token : profile.special_tokens) {
        if (token.empty()) {
            continue;
        }
        size_t pos = 0;
        while (true) {
            pos = text.find(token, pos);
            if (pos == std::string_view::npos) {
                break;
            }
            ++overhead;
            pos += token.size();
        }
    }
    return overhead;
}

size_t EstimateSegmentTokens(size_t segment_bytes, SegmentKind segment_kind) {
    if (segment_bytes == 0) {
        return 0;
    }

    constexpr double kWordBytesPerToken = 3.5;
    constexpr double kDigitBytesPerToken = 3.0;
    constexpr double kPunctBytesPerToken = 2.8;
    constexpr double kUnicodeBytesPerToken = 2.2;
    constexpr double kSymbolBytesPerToken = 4.0;
    constexpr double kWhitespaceBytesPerToken = 5.5;

    double bytes_per_token = 4.0;
    switch (segment_kind) {
        case SegmentKind::AsciiWord:
            bytes_per_token = kWordBytesPerToken;
            break;
        case SegmentKind::AsciiDigit:
            bytes_per_token = kDigitBytesPerToken;
            break;
        case SegmentKind::AsciiPunctuation:
            bytes_per_token = kPunctBytesPerToken;
            break;
        case SegmentKind::AsciiSymbol:
            bytes_per_token = kSymbolBytesPerToken;
            break;
        case SegmentKind::AsciiWhitespace:
            bytes_per_token = kWhitespaceBytesPerToken;
            break;
        case SegmentKind::Unicode:
            bytes_per_token = kUnicodeBytesPerToken;
            break;
        case SegmentKind::Control:
        default:
            break;
    }

    return static_cast<size_t>(std::max(1.0, std::ceil(segment_bytes / bytes_per_token)));
}

size_t ConservativeEstimate(const std::string& text, const TokenizerProfile& profile) {
    size_t total_tokens = 0;
    if (text.empty()) {
        return 0;
    }

    for (size_t i = 0; i < text.size();) {
        const uint8_t byte = static_cast<uint8_t>(text[i]);
        size_t segment_bytes = 0;
        SegmentKind segment_kind = SegmentKind::Control;

        if ((byte & 0x80) == 0) {
            segment_kind = ClassifyAscii(byte);
            while (i + segment_bytes < text.size()) {
                const uint8_t next_byte = static_cast<uint8_t>(text[i + segment_bytes]);
                if ((next_byte & 0x80) != 0) {
                    break;
                }
                if (ClassifyAscii(next_byte) != segment_kind) {
                    break;
                }
                segment_bytes++;
            }
        } else {
            segment_kind = SegmentKind::Unicode;
            segment_bytes = CountUtf8Bytes(text, i);
            // keep each UTF-8 sequence as its own conservative segment
            if (i + segment_bytes < text.size() && (static_cast<uint8_t>(text[i + segment_bytes]) & 0x80) != 0) {
                segment_kind = SegmentKind::Unicode;
                // merge adjacent non-ascii bytes conservatively by continuing this loop
                while (i + segment_bytes < text.size() &&
                       (static_cast<uint8_t>(text[i + segment_bytes]) & 0x80) != 0) {
                    segment_bytes += CountUtf8Bytes(text, i + segment_bytes);
                }
            }
        }

        total_tokens += EstimateSegmentTokens(segment_bytes, segment_kind);
        i += segment_bytes;
    }

    const double metadata_factor =
            1.0 + (std::min<size_t>(static_cast<size_t>(2000), profile.merge_count) / 2000.0 / 10.0);
    const size_t conservative_tokens = static_cast<size_t>(std::ceil(total_tokens * metadata_factor));
    const size_t special_token_overhead = CountSpecialTokenOverhead(text, profile);
    return std::max(total_tokens, conservative_tokens) + special_token_overhead;
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

    const auto& profile = DefaultTokenizerProfile();
    if (!profile.Loaded()) {
        throw std::runtime_error("Default tokenizer profile failed to initialize.");
    }
    return ConservativeEstimate(text, profile);
}

void PromptTokenizer::InitializeDefaultTokenizer() {
    (void)DefaultTokenizerProfile();
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
        throw std::runtime_error(
                BudgetExceededMessage(function_name, first_count_tokens, usable_context, model_details));
    }
    return best_count;
}

nlohmann::json PromptBatcher::SliceColumns(const nlohmann::json& columns, int start_index, int count) {
    nlohmann::json result = nlohmann::json::array();
    for (const auto& column : columns) {
        nlohmann::json output_column = nlohmann::json::object();
        for (const auto& item : column.items()) {
            if (item.key() == "data") {
                output_column["data"] = nlohmann::json::array();
                for (int row_idx = 0; row_idx < count && start_index + row_idx < static_cast<int>(item.value().size());
                     row_idx++) {
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
        for (const auto& column : source) {
            nlohmann::json output_column = nlohmann::json::object();
            for (const auto& item : column.items()) {
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
