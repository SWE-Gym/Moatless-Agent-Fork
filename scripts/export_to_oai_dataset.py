import argparse
import json
import os


def export_to_oai_dataset(exp_path, fail_only=False):
    evals = []
    with open(f"{exp_path}/preds.swebench_eval.jsonl") as f:
        for line in f:
            evals.append(json.loads(line))

    def get_success_instances(evals, repos=None, fail_only=False):
        if repos:
            evals = [dp for dp in evals if dp["instance_id"].split("__")[0] in repos]
        sucess_ones = [
            dp["instance_id"]
            for dp in evals
            if dp["test_result"]["report"]["resolved"] == (not fail_only)
        ]
        return sucess_ones, len(evals)

    success_ids, total_len = get_success_instances(evals, fail_only=fail_only)
    prompt_prefix = exp_path + "/prompt_logs"
    prompts = {}
    for id in success_ids:
        prompt_dir = f"{prompt_prefix}/{id}"
        prompt_files = os.listdir(prompt_dir)
        prompt_files = [pf for pf in prompt_files if pf.endswith(".json")]
        prompts[id] = {}
        for pf in prompt_files:
            with open(f"{prompt_dir}/{pf}") as f:
                prompts[id][pf] = json.load(f)

    def to_openai_list(prompts):
        openai_list = []
        for id, prompt in prompts.items():
            for pf, p in prompt.items():
                if p["error"]:
                    continue
                message = p["messages"]
                message.append(p["completion"][0])
                assert len(p["completion"]) == 1
                openai_list.append(
                    {
                        "messages": message,
                        "instance_id": id,
                        "exp_name": exp_path.split("/")[-1],
                        "fail": fail_only
                    }
                )
        return openai_list

    dataset = to_openai_list(prompts)
    print(
        f"Total instances: {total_len}, success instances: {len(success_ids)}, dataset size: {len(dataset)}"
    )
    output_path = f"{exp_path}/dataset.openai.jsonl"
    if fail_only:
        output_path = f"{exp_path}/dataset.openai.fail.jsonl"
    with open(output_path, "w") as f:
        for dp in dataset:
            f.write(json.dumps(dp) + "\n")
    print(f"Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path", type=str)
    parser.add_argument("--fail-only", action="store_true")
    args = parser.parse_args()
    export_to_oai_dataset(args.exp_path, args.fail_only)

if __name__ == "__main__":
    main()