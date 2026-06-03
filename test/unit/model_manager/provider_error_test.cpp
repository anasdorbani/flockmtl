#include "flock/model_manager/providers/handlers/ollama.hpp"
#include "flock/model_manager/providers/handlers/openai.hpp"
#include "nlohmann/json.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace flock {

class TestOpenAIModelManager : public OpenAIModelManager {
public:
    TestOpenAIModelManager() : OpenAIModelManager("test-token", "", true) {}

    using BaseModelProviderHandler::checkResponse;
};

class TestOllamaModelManager : public OllamaModelManager {
public:
    TestOllamaModelManager() : OllamaModelManager("http://localhost:11434", true) {}

    using BaseModelProviderHandler::checkResponse;
};

TEST(ProviderErrorTest, JsonErrorMessageIsSurfacedDirectly) {
    TestOpenAIModelManager manager;
    nlohmann::json response = {
            {"error", {
                              {"type", "invalid_request_error"},
                              {"message", "This model's maximum context length was exceeded"}}}};

    try {
        manager.checkResponse(response, IModelProviderHandler::RequestType::Completion);
        FAIL() << "Expected provider error";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "This model's maximum context length was exceeded");
        EXPECT_THAT(error.what(), ::testing::Not(::testing::HasSubstr("Provider error")));
        EXPECT_THAT(error.what(), ::testing::Not(::testing::HasSubstr("Response processing error")));
    }
}

TEST(ProviderErrorTest, OpenAIUnfinishedResponseUsesFallbackReason) {
    TestOpenAIModelManager manager;
    nlohmann::json response = {
            {"choices", nlohmann::json::array({
                                {{"finish_reason", "length"},
                                 {"message", {{"content", "{\"items\":[\"partial\"]}"}}}}})}};

    try {
        manager.checkResponse(response, IModelProviderHandler::RequestType::Completion);
        FAIL() << "Expected unfinished response error";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "Model response did not finish successfully. finish_reason: length");
        EXPECT_THAT(error.what(), ::testing::Not(::testing::HasSubstr("Response processing error")));
    }
}

TEST(ProviderErrorTest, OllamaUnfinishedResponseUsesProviderMessageWhenAvailable) {
    TestOllamaModelManager manager;
    nlohmann::json response = {
            {"done_reason", "load"},
            {"message", "model returned an unfinished response"}};

    try {
        manager.checkResponse(response, IModelProviderHandler::RequestType::Completion);
        FAIL() << "Expected unfinished response error";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "model returned an unfinished response");
        EXPECT_THAT(error.what(), ::testing::Not(::testing::HasSubstr("done_reason")));
    }
}

}// namespace flock
