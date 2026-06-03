#include "flock/functions/token_budget.hpp"
#include <gtest/gtest.h>
#include <string>

namespace flock {

TEST(PromptTokenizerTest, CountsNonEmptyPrompt) {
    PromptTokenizer::SetTokenCounterForTesting(nullptr);

    EXPECT_GT(PromptTokenizer::CountTokens("hello world"), 0);
}

TEST(PromptBatcherTest, FindsLargestTupleCountWithinBudget) {
    ModelDetails details;
    details.batch_size = 16;
    details.context_window = 4;
    details.safe_margin = 1;

    PromptTokenizer::SetTokenCounterForTesting([](const std::string& prompt) {
        return std::stoul(prompt);
    });

    const auto count = PromptBatcher::FindMaxTupleCount(
            details, 5,
            [](int tuple_count) {
                return std::to_string(tuple_count);
            },
            "test_llm");

    EXPECT_EQ(count, 3);
    PromptTokenizer::SetTokenCounterForTesting(nullptr);
}

TEST(PromptBatcherTest, OversizedSingleTupleFailsClearly) {
    ModelDetails details;
    details.batch_size = 16;
    details.context_window = 4;
    details.safe_margin = 1;

    PromptTokenizer::SetTokenCounterForTesting([](const std::string&) {
        return 4;
    });

    EXPECT_THROW(
            PromptBatcher::FindMaxTupleCount(
                    details, 1,
                    [](int) {
                        return "prompt";
                    },
                    "test_llm"),
            std::runtime_error);
    PromptTokenizer::SetTokenCounterForTesting(nullptr);
}

}// namespace flock
