import os
import json

os.environ["VOYAGE_API_KEY"] = "<VOYAGE_API>"
os.environ["OPENAI_API_KEY"] = (
    "<OPENAI_API_KEY>"
)
os.environ["OPENROUTER_API_KEY"] = (
    "<OPENROUTER_API_KEY>"
)
os.environ["ANTHROPIC_API_KEY"] = ""

from moatless.edit import EditCode, PlanToCode
from moatless.find import DecideRelevance, IdentifyCode, SearchCode
import datetime
import os
from instructor import Mode
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to configure model settings.")
    parser.add_argument("--instance", type=str, help="Instance to evaluate on")
    parser.add_argument(
        "--model", type=str, default="openai/yolo-sft", help="Model name or path"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature parameter for model generation",
    )
    parser.add_argument(
        "--max_cost", type=float, default=1.0, help="Maximum cost parameter"
    )
    parser.add_argument(
        "--serve_api_base", type=str, default=None, help="API base URL to serve model"
    )
    parser.add_argument(
        "--index_store_dir",
        type=str,
        default="/home/jiayipan/code/24FA/temp/ml-01/exp/index",
        help="Directory path for index storage",
    )
    parser.add_argument(
        "--note", type=str, default="N/A", help="Note for the evaluation"
    )
    parser.add_argument("--eval_dir", type=str)
    parser.add_argument("--eval_name", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str)
    args = parser.parse_args()
    # Convert args to a dictionary and unpack into individual variables
    config = vars(args)
    instructor_mode = Mode.JSON
    config.update({"instructor_mode": str(instructor_mode)})
    model = config["model"]
    temperature = config["temperature"]
    max_cost = config["max_cost"]
    serve_api_base = config["serve_api_base"]
    index_store_dir = config["index_store_dir"]
    note = config["note"]
    workers = 1

    global_params = {
        "model": model,
        "temperature": temperature,
        "max_tokens": 2000,
        "max_prompt_file_tokens": 12000,
    }
    # instructor_mode = None
    if serve_api_base:
        global_params["serve_api_base"] = serve_api_base
        os.environ["SERVE_API_BASE"] = global_params["serve_api_base"]
    os.environ["NUM_SAMPLING_WORKERS"] = str(workers)

    evaluations_dir = config["eval_dir"]
    evaluation_name = config["eval_name"]

    state_params = {
        SearchCode: {
            "max_search_results": 75,
            "provide_initial_context": True,  # Do a vector search with the problem statement to get an initial file context
            "initial_context_tokens": 6000,
            "initial_search_results": 100,
            "initial_context_spans_per_file": 5,
        },
        IdentifyCode: {
            "expand_context": True,  # Expands the search results with related code to the search hits
        },
        DecideRelevance: {
            "finish_after_relevant_count": 1,  # Even if the LLM doesn't believe the identified code is complete we will finish up after one retry
        },
        PlanToCode: {
            "max_tokens_in_edit_prompt": 750,  # The max number of tokens in the edit block
            "expand_context_with_related_spans": False,
            "finish_on_review": True,  # To abort if the LLm suggest reviews of the code, it's only possible to apply changes ATM.
        },
        EditCode: {
            "chain_of_thought": False,
            "show_file_context": False,
            "max_prompt_file_tokens": 8000,
        },
    }

    date_str = datetime.datetime.now().strftime("%Y%m%d")
    model_file_name = f"{model.replace('/', '_')}"

    # evaluation_name = f"{date_str}_moatless_{model_file_name}_temp_{temperature}"
    evaluation_dir = f"{evaluations_dir}/{evaluation_name}"
    trajectory_dir = f"{evaluations_dir}/{evaluation_name}/trajs"
    predictions_path = f"{evaluation_dir}/all_preds.jsonl"

    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)

    print(evaluation_dir)
    with open(f"{evaluation_dir}/args.json", "w") as f:
        json.dump(config, f)

    # %% [markdown]
    # ## Run the evaluation
    #
    # Test if evaluation works with a sub set of 5 instances. Remove this to run the full benchmark.

    # %% [markdown]
    # Run the evaluation

    # %%
    from moatless.transitions import search_and_code_transitions
    from moatless.benchmark.evaluation import Evaluation

    search_and_code = search_and_code_transitions(
        global_params=global_params, state_params=state_params
    )
    evaluation = Evaluation(
        transitions=search_and_code,
        evaluations_dir=evaluations_dir,
        evaluation_name=evaluation_name,
        index_store_dir=index_store_dir,
        repo_base_dir="/home/jiayipan/code/24FA/temp/ml-01/exp/repos",
        max_cost=max_cost,
        max_file_context_tokens=16000,
        instructor_mode=instructor_mode,
    )
    evaluation.run_swebench_evaluation(
        dataset=args.dataset,
        split=args.split,
        instance_ids=[args.instance],
    )
