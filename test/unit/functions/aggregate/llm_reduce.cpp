#include "flock/functions/aggregate/llm_reduce.hpp"
#include "llm_aggregate_function_test_base.hpp"

namespace flock {

class LLMReduceTest : public LLMAggregateTestBase<LlmReduce> {
protected:
    static constexpr const char* EXPECTED_RESPONSE = "A comprehensive summary of running shoes, wireless headphones, and smart watches, featuring advanced technology and user-friendly designs for active lifestyles.";

    std::string GetExpectedResponse() const override {
        return EXPECTED_RESPONSE;
    }

    nlohmann::json GetExpectedJsonResponse() const override {
        return nlohmann::json{{"items", {EXPECTED_RESPONSE}}};
    }

    std::string GetFunctionName() const override {
        return "llm_reduce";
    }

    AggregateFunctionType GetFunctionType() const override {
        return AggregateFunctionType::REDUCE;
    }

    nlohmann::json PrepareExpectedResponseForBatch(const std::vector<std::string>& responses) const override {
        return nlohmann::json{{"items", {responses[0]}}};
    }

    nlohmann::json PrepareExpectedResponseForLargeInput(size_t input_count) const override {
        return nlohmann::json{{"items", {"Summary of " + std::to_string(input_count) + " items processed"}}};
    }

    std::string FormatExpectedResult(const nlohmann::json& response) const override {
        if (response.contains("items")) {
            return response["items"][0].get<std::string>();
        }
        return response.get<std::string>();
    }
};

// Test llm_reduce with SQL queries without GROUP BY - new API
TEST_F(LLMReduceTest, LLMReduceWithoutGroupBy) {
    EXPECT_CALL(*mock_provider, AddCompletionRequest(::testing::_, ::testing::_, ::testing::_, ::testing::_))
            .Times(1);
    EXPECT_CALL(*mock_provider, CollectCompletions(::testing::_))
            .WillOnce(::testing::Return(std::vector<nlohmann::json>{GetExpectedJsonResponse()}));

    auto con = Config::GetConnection();

    const auto results = con.Query(
            "SELECT " + GetFunctionName() + "("
                                            "{'model_name': 'gpt-4o'}, "
                                            "{'prompt': 'Summarize the following product descriptions', 'context_columns': [{'data': description}]}"
                                            ") AS product_summary FROM VALUES "
                                            "('High-performance running shoes with advanced cushioning'), "
                                            "('Wireless noise-cancelling headphones for immersive audio'), "
                                            "('Smart fitness tracker with heart rate monitoring') AS products(description);");

    ASSERT_EQ(results->RowCount(), 1);
    ASSERT_EQ(results->GetValue(0, 0).GetValue<std::string>(), GetExpectedResponse());
}

// Test llm_reduce with SQL queries with GROUP BY - new API
TEST_F(LLMReduceTest, LLMReduceWithGroupBy) {
    EXPECT_CALL(*mock_provider, AddCompletionRequest(::testing::_, ::testing::_, ::testing::_, ::testing::_))
            .Times(3);
    EXPECT_CALL(*mock_provider, CollectCompletions(::testing::_))
            .Times(3)
            .WillRepeatedly(::testing::Return(std::vector<nlohmann::json>{GetExpectedJsonResponse()}));

    auto con = Config::GetConnection();

    const auto results = con.Query(
            "SELECT category, " + GetFunctionName() + "("
                                                      "{'model_name': 'gpt-4o'}, "
                                                      "{'prompt': 'Summarize the following product descriptions', 'context_columns': [{'data': description}]}"
                                                      ") AS description_summary FROM VALUES "
                                                      "('electronics', 'High-performance running shoes with advanced cushioning'), "
                                                      "('audio', 'Wireless noise-cancelling headphones for immersive audio'), "
                                                      "('fitness', 'Smart fitness tracker with heart rate monitoring') "
                                                      "AS products(category, description) GROUP BY category;");

    ASSERT_EQ(results->RowCount(), 3);
    ASSERT_EQ(results->GetValue(1, 0).GetValue<std::string>(), GetExpectedResponse());
    ASSERT_EQ(results->GetValue(1, 1).GetValue<std::string>(), GetExpectedResponse());
    ASSERT_EQ(results->GetValue(1, 2).GetValue<std::string>(), GetExpectedResponse());
}

// Test argument validation
TEST_F(LLMReduceTest, ValidateArguments) {
    TestValidateArguments();
}

// Test operation with invalid arguments
TEST_F(LLMReduceTest, Operation_InvalidArguments_ThrowsException) {
    TestOperationInvalidArguments();
}

// Test operation with multiple input scenarios - new API
TEST_F(LLMReduceTest, Operation_MultipleInputs_ProcessesCorrectly) {
    const nlohmann::json expected_response = GetExpectedJsonResponse();

    EXPECT_CALL(*mock_provider, AddCompletionRequest(::testing::_, ::testing::_, ::testing::_, ::testing::_))
            .Times(3);
    EXPECT_CALL(*mock_provider, CollectCompletions(::testing::_))
            .Times(3)
            .WillRepeatedly(::testing::Return(std::vector<nlohmann::json>{expected_response}));

    auto con = Config::GetConnection();

    const auto results = con.Query(
            "SELECT name, " + GetFunctionName() + "("
                                                  "{'model_name': 'gpt-4o'}, "
                                                  "{'prompt': 'Summarize the following product information', 'context_columns': [{'data': name}, {'data': description}]}"
                                                  ") AS comprehensive_summary FROM VALUES "
                                                  "('Running Shoes', 'High-performance running shoes with advanced cushioning'), "
                                                  "('Headphones', 'Wireless noise-cancelling headphones for immersive audio'), "
                                                  "('Fitness Tracker', 'Smart fitness tracker with heart rate monitoring') "
                                                  "AS products(name, description) GROUP BY name;");

    ASSERT_EQ(results->RowCount(), 3);
    ASSERT_EQ(results->GetValue(1, 0).GetValue<std::string>(), GetExpectedResponse());
    ASSERT_EQ(results->GetValue(1, 1).GetValue<std::string>(), GetExpectedResponse());
    ASSERT_EQ(results->GetValue(1, 2).GetValue<std::string>(), GetExpectedResponse());
}

// Test large input set processing - new API
TEST_F(LLMReduceTest, Operation_LargeInputSet_ProcessesCorrectly) {
    constexpr size_t input_count = 100;
    const nlohmann::json expected_response = PrepareExpectedResponseForLargeInput(input_count);

    EXPECT_CALL(*mock_provider, AddCompletionRequest(::testing::_, ::testing::_, ::testing::_, ::testing::_))
            .Times(100);
    EXPECT_CALL(*mock_provider, CollectCompletions(::testing::_))
            .Times(100)
            .WillRepeatedly(::testing::Return(std::vector<nlohmann::json>{expected_response}));

    auto con = Config::GetConnection();

    const auto results = con.Query(
            "SELECT id, " + GetFunctionName() + "("
                                                "{'model_name': 'gpt-4o'}, "
                                                "{'prompt': 'Summarize all product descriptions', 'context_columns': [{'data': 'Product description ' || id::TEXT}]}"
                                                ") AS large_summary FROM range(" +
            std::to_string(input_count) + ") AS t(id) GROUP BY id;");

    ASSERT_EQ(results->RowCount(), 100);
    for (size_t i = 0; i < input_count; i++) {
        ASSERT_EQ(results->GetValue(1, i).GetValue<std::string>(), FormatExpectedResult(expected_response));
    }
}

}// namespace flock
