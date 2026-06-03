#include "flock/functions/token_budget.hpp"
#include <gtest/gtest.h>
#include <string>

namespace flock {

TEST(PromptTokenizerTest, CountsNonEmptyPrompt) {
    PromptTokenizer::SetTokenCounterForTesting(nullptr);

    EXPECT_GT(PromptTokenizer::CountTokens("hello world"), 0);
}

TEST(PromptTokenizerTest, CountTokensIsMonotonicForLongerPrompt) {
    PromptTokenizer::SetTokenCounterForTesting(nullptr);

    const auto shorter_count = PromptTokenizer::CountTokens("the quick brown fox");
    const auto longer_count = PromptTokenizer::CountTokens("the quick brown fox the quick brown fox");
    EXPECT_GE(longer_count, shorter_count);
}

TEST(PromptTokenizerTest, CountTokensIsDeterministic) {
    PromptTokenizer::SetTokenCounterForTesting(nullptr);

    const std::string prompt = "some text with 🎉 emoji and symbols !@# and numbers 12345";
    const auto first = PromptTokenizer::CountTokens(prompt);
    const auto second = PromptTokenizer::CountTokens(prompt);

    EXPECT_EQ(first, second);
}

TEST(PromptTokenizerTest, CountTokensHandlesComplexUnicodeAndControlChars) {
    PromptTokenizer::SetTokenCounterForTesting(nullptr);

    const std::string control_chars = std::string("line 1\nline 2\twith ") + std::string(1, '\0') + "control";
    EXPECT_NO_THROW({
        const auto count = PromptTokenizer::CountTokens(control_chars);
        EXPECT_GT(count, 0);
    });
    const auto repeated = PromptTokenizer::CountTokens(std::string(500, 'x'));
    EXPECT_GT(repeated, 0);
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

TEST(PromptBatcherTest, SplitsOnContextBeforeBatchSize) {
    ModelDetails details;
    details.batch_size = 32;
    details.context_window = 8;
    details.safe_margin = 2;

    PromptTokenizer::SetTokenCounterForTesting([](const std::string& prompt) {
        return std::stoul(prompt);
    });

    const auto count = PromptBatcher::FindMaxTupleCount(
            details, 10,
            [](int tuple_count) {
                return std::to_string(tuple_count);
            },
            "test_llm");
    EXPECT_EQ(count, 6);
    PromptTokenizer::SetTokenCounterForTesting(nullptr);
}

TEST(PromptBatcherTest, ContextBudgetFailureIncludesTupleMessage) {
    ModelDetails details;
    details.batch_size = 16;
    details.context_window = 4;
    details.safe_margin = 1;

    PromptTokenizer::SetTokenCounterForTesting([](const std::string&) {
        return 4;
    });
    try {
        PromptBatcher::FindMaxTupleCount(
                details, 1,
                [](int) {
                    return "prompt";
                },
                "test_llm");
        FAIL() << "Expected an exception for oversized tuple";
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("single tuple requires"), std::string::npos);
        EXPECT_NE(message.find("safe_margin"), std::string::npos);
    }
    PromptTokenizer::SetTokenCounterForTesting(nullptr);
}

}// namespace flock
