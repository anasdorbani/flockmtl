#include "flock/functions/scalar/scalar.hpp"
#include "flock/functions/token_budget.hpp"
#include "flock/model_manager/model.hpp"
#include <duckdb/planner/expression/bound_function_expression.hpp>

namespace flock {

void ScalarFunctionBase::ValidateArgumentCount(
        const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& arguments,
        const std::string& function_name) {
    if (arguments.size() != 2) {
        throw duckdb::BinderException(
                function_name + " requires 2 arguments: (1) model, (2) prompt with context_columns. Got " +
                std::to_string(arguments.size()));
    }
}

void ScalarFunctionBase::ValidateArgumentTypes(
        const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& arguments,
        const std::string& function_name) {
    if (arguments[0]->return_type.id() != duckdb::LogicalTypeId::STRUCT) {
        throw duckdb::BinderException(function_name + ": First argument must be model (struct type)");
    }
    if (arguments[1]->return_type.id() != duckdb::LogicalTypeId::STRUCT) {
        throw duckdb::BinderException(
                function_name + ": Second argument must be prompt with context_columns (struct type)");
    }
}

ScalarFunctionBase::PromptStructInfo ScalarFunctionBase::ExtractPromptStructInfo(
        const duckdb::LogicalType& prompt_type) {
    PromptStructInfo info{false, std::nullopt, ""};

    for (idx_t i = 0; i < duckdb::StructType::GetChildCount(prompt_type); i++) {
        auto field_name = duckdb::StructType::GetChildName(prompt_type, i);
        if (field_name == "context_columns") {
            info.has_context_columns = true;
        } else if (field_name == "prompt" || field_name == "prompt_name") {
            if (!info.prompt_field_index.has_value()) {
                info.prompt_field_index = i;
                info.prompt_field_name = field_name;
            }
        }
    }

    return info;
}

void ScalarFunctionBase::ValidatePromptStructFields(const PromptStructInfo& info,
                                                    const std::string& function_name,
                                                    bool require_context_columns) {
    if (require_context_columns && !info.has_context_columns) {
        throw duckdb::BinderException(
                function_name + ": Second argument must contain 'context_columns' field");
    }
}

void ScalarFunctionBase::InitializeModelJson(
        duckdb::ClientContext& context,
        const duckdb::unique_ptr<duckdb::Expression>& model_expr,
        LlmFunctionBindData& bind_data) {
    if (!model_expr->IsFoldable()) {
        return;
    }

    auto model_value = duckdb::ExpressionExecutor::EvaluateScalar(context, *model_expr);
    auto user_model_json = CastValueToJson(model_value);
    bind_data.model_json = Model::ResolveModelDetailsToJson(user_model_json);
}

nlohmann::json ScalarFunctionBase::Complete(nlohmann::json& columns, const std::string& user_prompt,
                                            ScalarFunctionType function_type, Model& model) {
    const auto [prompt, media_data] = PromptManager::Render(user_prompt, columns, function_type, model.GetModelDetails().tuple_format);
    OutputType output_type = OutputType::STRING;
    if (function_type == ScalarFunctionType::FILTER) {
        output_type = OutputType::BOOL;
    }

    model.AddCompletionRequest(prompt, static_cast<int>(columns[0]["data"].size()), output_type, media_data);
    auto response = model.CollectCompletions();
    return response[0]["items"];
};

nlohmann::json ScalarFunctionBase::BatchAndComplete(const nlohmann::json& tuples,
                                                    const std::string& user_prompt,
                                                    const ScalarFunctionType function_type, Model& model) {
    const auto model_details = model.GetModelDetails();
    const auto render_tuples = PromptManager::PrepareColumnsForRender(tuples);
    auto responses = nlohmann::json::array();

    int start_index = 0;
    const auto num_tuples = PromptBatcher::RowCount(render_tuples);

    while (start_index < num_tuples) {
        const auto max_batch_size = PromptBatcher::EffectiveBatchSize(model_details, num_tuples - start_index);
        const auto batch_size = PromptBatcher::FindMaxTupleCount(
                model_details, max_batch_size,
                [&](int tuple_count) {
                    auto candidate = PromptBatcher::SliceColumns(render_tuples, start_index, tuple_count);
                    const auto [prompt, media_data] = PromptManager::Render(
                            user_prompt, candidate, function_type, model_details.tuple_format);
                    (void)media_data;
                    return prompt;
                },
                function_type == ScalarFunctionType::FILTER ? "llm_filter" : "llm_complete");

        auto batch_tuples = PromptBatcher::SliceColumns(render_tuples, start_index, batch_size);
        auto response = Complete(batch_tuples, user_prompt, function_type, model);
        const auto expected_response_count = batch_tuples[0]["data"].size();

        if (response.size() < expected_response_count) {
            for (auto i = static_cast<int>(response.size()); i < static_cast<int>(expected_response_count); i++) {
                response.push_back(nullptr);
            }
        } else if (response.size() > expected_response_count) {
            response.erase(response.begin() + expected_response_count, response.end());
        }

        for (const auto& tuple: response) {
            responses.push_back(tuple);
        }

        start_index += batch_size;
    }

    return responses;
}

void ScalarFunctionBase::InitializePrompt(
        duckdb::ClientContext& context,
        const duckdb::unique_ptr<duckdb::Expression>& prompt_expr,
        LlmFunctionBindData& bind_data) {
    nlohmann::json prompt_json;

    if (prompt_expr->IsFoldable()) {
        auto prompt_value = duckdb::ExpressionExecutor::EvaluateScalar(context, *prompt_expr);
        prompt_json = CastValueToJson(prompt_value);
    } else if (prompt_expr->expression_class == duckdb::ExpressionClass::BOUND_FUNCTION) {
        auto& func_expr = prompt_expr->Cast<duckdb::BoundFunctionExpression>();
        const auto& struct_type = prompt_expr->return_type;

        for (idx_t i = 0; i < duckdb::StructType::GetChildCount(struct_type) && i < func_expr.children.size(); i++) {
            auto field_name = duckdb::StructType::GetChildName(struct_type, i);
            auto& child = func_expr.children[i];

            if (field_name != "context_columns" && child->IsFoldable()) {
                try {
                    auto field_value = duckdb::ExpressionExecutor::EvaluateScalar(context, *child);
                    if (field_value.type().id() == duckdb::LogicalTypeId::VARCHAR) {
                        prompt_json[field_name] = field_value.GetValue<std::string>();
                    } else {
                        prompt_json[field_name] = CastValueToJson(field_value);
                    }
                } catch (...) {
                    // Skip fields that can't be evaluated
                }
            }
        }
    }

    if (prompt_json.contains("context_columns")) {
        prompt_json.erase("context_columns");
    }

    auto prompt_details = PromptManager::CreatePromptDetails(prompt_json);
    bind_data.prompt = prompt_details.prompt;
}

duckdb::unique_ptr<LlmFunctionBindData> ScalarFunctionBase::ValidateAndInitializeBindData(
        duckdb::ClientContext& context,
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& arguments,
        const std::string& function_name,
        bool require_context_columns,
        bool initialize_prompt) {

    ValidateArgumentCount(arguments, function_name);
    ValidateArgumentTypes(arguments, function_name);

    const auto& prompt_type = arguments[1]->return_type;
    auto prompt_info = ExtractPromptStructInfo(prompt_type);
    ValidatePromptStructFields(prompt_info, function_name, require_context_columns);

    auto bind_data = duckdb::make_uniq<LlmFunctionBindData>();

    InitializeModelJson(context, arguments[0], *bind_data);
    if (initialize_prompt) {
        InitializePrompt(context, arguments[1], *bind_data);
    }

    return bind_data;
}

}// namespace flock
