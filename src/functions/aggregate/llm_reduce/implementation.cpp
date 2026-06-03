#include "flock/core/config.hpp"
#include "flock/functions/aggregate/llm_reduce.hpp"
#include "flock/functions/llm_function_bind_data.hpp"
#include "flock/functions/token_budget.hpp"
#include "flock/metrics/manager.hpp"

#include <chrono>

namespace flock {

duckdb::unique_ptr<duckdb::FunctionData> LlmReduce::Bind(
        duckdb::ClientContext& context,
        duckdb::AggregateFunction& function,
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& arguments) {
    return AggregateFunctionBase::ValidateAndInitializeBindData(context, arguments, "llm_reduce");
}

nlohmann::json LlmReduce::ReduceBatch(nlohmann::json& tuples,
                                      const AggregateFunctionType& function_type,
                                      const nlohmann::json& summary) {
    auto [prompt, media_data] = PromptManager::Render(
            user_query, tuples, function_type, model.GetModelDetails().tuple_format);

    prompt += "\n\n" + summary.dump(4);

    model.AddCompletionRequest(prompt, 1, OutputType::STRING, media_data);
    auto response = model.CollectCompletions()[0];
    return response["items"][0];
}

nlohmann::json LlmReduce::ReduceLoop(const nlohmann::json& tuples,
                                     const AggregateFunctionType& function_type) {
    auto batch_tuples = nlohmann::json::array();
    auto summary = nlohmann::json::object({{"Previous Batch Summary", ""}});
    int start_index = 0;
    const auto render_tuples = PromptManager::PrepareColumnsForRender(tuples);
    int num_tuples = PromptBatcher::RowCount(render_tuples);
    const auto model_details = model.GetModelDetails();

    while (start_index < num_tuples) {
        const auto max_batch_size = PromptBatcher::EffectiveBatchSize(model_details, num_tuples - start_index);
        const auto batch_size = PromptBatcher::FindMaxTupleCount(
                model_details, max_batch_size,
                [&](int tuple_count) {
                    auto candidate = PromptBatcher::SliceColumns(render_tuples, start_index, tuple_count);
                    auto [prompt, media_data] = PromptManager::Render(
                            user_query, candidate, function_type, model_details.tuple_format);
                    (void)media_data;
                    prompt += "\n\n" + summary.dump(4);
                    return prompt;
                },
                "llm_reduce");

        batch_tuples = PromptBatcher::SliceColumns(render_tuples, start_index, batch_size);
        start_index += batch_size;

        auto response = ReduceBatch(batch_tuples, function_type, summary);
        batch_tuples.clear();
        summary = nlohmann::json::object({{"Previous Batch Summary", response}});
    }

    return summary["Previous Batch Summary"];
}

void LlmReduce::FinalizeResults(duckdb::Vector& states, duckdb::AggregateInputData& aggr_input_data,
                                duckdb::Vector& result, idx_t count, idx_t offset,
                                const AggregateFunctionType function_type) {
    const auto states_vector = reinterpret_cast<AggregateFunctionState**>(
            duckdb::FlatVector::GetData<duckdb::data_ptr_t>(states));

    // Get bind data - model_json and prompt are guaranteed to be initialized
    auto& bind_data = aggr_input_data.bind_data->Cast<LlmFunctionBindData>();

    // Get model details for metrics (create temp model just for details)
    auto temp_model = bind_data.CreateModel();
    auto model_details_obj = temp_model.GetModelDetails();

    auto db = Config::db;
    std::vector<const void*> processed_state_ids;

    // Process each state individually
    for (idx_t i = 0; i < count; i++) {
        auto result_idx = i + offset;
        auto* state = states_vector[i];

        if (!state || !state->value || state->value->empty()) {
            result.SetValue(result_idx, nullptr);
            continue;
        }

        // Track metrics for this state
        const void* state_id = static_cast<const void*>(state);
        processed_state_ids.push_back(state_id);
        MetricsManager::StartInvocation(db, state_id, FunctionType::LLM_REDUCE);
        MetricsManager::SetModelInfo(model_details_obj.model_name, model_details_obj.provider_name);

        auto exec_start = std::chrono::high_resolution_clock::now();

        // Create function instance with bind data and process
        // IMPORTANT: Use CreateModel() for thread-safe Model instance
        LlmReduce reduce_instance;
        reduce_instance.model = bind_data.CreateModel();
        reduce_instance.user_query = bind_data.prompt;
        auto response = reduce_instance.ReduceLoop(*state->value, function_type);

        auto exec_end = std::chrono::high_resolution_clock::now();
        double exec_duration_ms = std::chrono::duration<double, std::milli>(exec_end - exec_start).count();
        MetricsManager::AddExecutionTime(exec_duration_ms);

        if (response.is_string()) {
            result.SetValue(result_idx, response.get<std::string>());
        } else {
            result.SetValue(result_idx, response.dump());
        }
    }

    // Merge all metrics from processed states
    if (!processed_state_ids.empty()) {
        MetricsManager::MergeAggregateMetrics(db, processed_state_ids, FunctionType::LLM_REDUCE,
                                              model_details_obj.model_name, model_details_obj.provider_name);
    }
}

}// namespace flock
