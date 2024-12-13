from moatless.benchmark.swebench import load_instances
from moatless.repository import FileRepository
import os
import json


from moatless.benchmark.swebench import setup_swebench_repo


def next_instance(instances):
    if not instances:
        return None
    instance = instances.pop(0)
    print(f"Instance: {instance['instance_id']}, {len(instances)} instances left")
    return instance


from moatless.index.settings import IndexSettings
from moatless.index.code_index import CodeIndex
from dotenv import load_dotenv
from moatless.benchmark.swebench import get_repo_dir_name
import os
os.environ["VOYAGE_API_KEY"] = "<VOYAGE_API_KEY>"
INDEX_PATH = "/home/jiayipan/code/24FA/temp/ml-01/exp/index"

def get_persist_dir(instance):
    return os.path.join(INDEX_PATH, get_repo_dir_name(instance["instance_id"]))

def print_previous_instances(previous_instances):
    for repo, instance in previous_instances.items():
        print(f"Repo: {repo}, Instance: {instance['instance_id']}")

def index(instance, previous_instances):
    print("---")
    if os.path.exists(get_persist_dir(instance)):
        print(f"Index for {instance['instance_id']} already exists. Skipping.")
        previous_instances[instance["repo"]] = instance
        return
    index_settings = IndexSettings(embed_model="voyage-code-2")
    repo_path = setup_swebench_repo(instance, repo_base_dir="/home/jiayipan/code/24FA/temp/ml-01/moatless-tools/t/repos", shared_dir=True)
    print(f"Repo path: {repo_path}")
    file_repo = FileRepository(repo_path)
    previous_instance = previous_instances.get(instance["repo"])
    print(f"Previous instance: {previous_instance['instance_id'] if previous_instance else None}")
    if previous_instance:
        print(f"Loading cache index from {previous_instance['instance_id']}")
        code_index =  CodeIndex.from_persist_dir(get_persist_dir(previous_instance), file_repo=file_repo)
    else:
        print(f"No cache index found. Building new index.")
        code_index = CodeIndex(settings=index_settings, file_repo=file_repo)

    print(f"Repo path: {repo_path}")

    vectors, indexed_tokens = code_index.run_ingestion(num_workers=2)
    print(f"Indexed {vectors} vectors and {indexed_tokens} tokens.")
    
    persist_dir = get_persist_dir(instance)
    code_index.persist(persist_dir=persist_dir)
    print(f"Index persisted to {persist_dir}")
    
    previous_instances[instance["repo"]] = instance

def index_all_instance(instances):
    previous_instances = {}
    instance = next_instance(instances)
    while instance:
        print_previous_instances(previous_instances)
        index(instance, previous_instances)
        instance = next_instance(instances)

def main():
    instance_by_id = load_instances("princeton-nlp/SWE-bench_Verified", split="test")
    instances = list(instance_by_id.values())
    # verified_100 = open("/home/jiayipan/code/24FA/temp/ml-01/moatless-tools/scripts/splits/verified_100.txt").read().splitlines()
    verified_50 = open("/home/jiayipan/code/24FA/temp/ml-01/moatless-tools/scripts/splits/verified_50.txt").read().splitlines()
    instances = [instance for instance in instances if instance["instance_id"] in verified_50]
    print(f"Number of instances: {len(instances)}")
    instances = sorted(instances, key=lambda x: x["created_at"])
    print(f"Number of instances: {len(instances)}")
    index_all_instance(instances)

if __name__ == "__main__":
    main()