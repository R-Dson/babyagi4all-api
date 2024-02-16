import os
import time
import logging
from collections import deque
from typing import Dict, List
import chromadb
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import InstructorEmbeddingFunction
import requests
import ooba_web
import re

# Load default environment variables (.env)
load_dotenv()

# Table config
RESULTS_STORE_NAME = os.getenv("RESULTS_STORE_NAME", os.getenv("TABLE_NAME", ""))
assert RESULTS_STORE_NAME, "\033[91m\033[1m" + "RESULTS_STORE_NAME environment variable is missing from .env" + "\033[0m\033[0m"

# Run configuration
INSTANCE_NAME = os.getenv("INSTANCE_NAME", os.getenv("BABY_NAME", "BabyAGI"))
COOPERATIVE_MODE = "none"
JOIN_EXISTING_OBJECTIVE = False

# Model configuration
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 256))
MAX_TASKS = int(os.getenv("MAX_TASKS", 10))

VERBOSE = (os.getenv("VERBOSE", "false").lower() == "true")
CTX_MAX = 16384

OOBA_API_HOST = os.getenv("OOBA_API_HOST")
OOBA_API_PORT = os.getenv("OOBA_API_PORT")

OLLAMA_API_HOST = os.getenv("OLLAMA_API_HOST")
OLLAMA_API_PORT = os.getenv("OLLAMA_API_PORT")

USE_OLLAMA = (os.getenv("USE_OLLAMA", "false").lower() == "true")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# Goal configuation
OBJECTIVE = os.getenv("OBJECTIVE", "")
OBJECTIVE_SPLIT_TASK = f"""Develop a concise list of essential tasks to complete in order to attain the objective."""
#OBJECTIVE_SPLIT_TASK = f"Develop a list of essential tasks to complete in order to attain the objective. Ensure each task contributes directly to achieving the objective and that the list is as concise as possible."

print("\033[95m\033[1m"+"\n*****CONFIGURATION*****\n"+"\033[0m\033[0m")
print(f"Name  : {INSTANCE_NAME}")

# Check if we know what we are doing
assert OBJECTIVE, "\033[91m\033[1m" + "OBJECTIVE environment variable is missing from .env" + "\033[0m\033[0m"

print("\033[94m\033[1m" + "\n*****OBJECTIVE*****\n" + "\033[0m\033[0m")
print(f"{OBJECTIVE}")

if not JOIN_EXISTING_OBJECTIVE: print("\033[93m\033[1m" + "\nInitial task:" + "\033[0m\033[0m" + f" {OBJECTIVE_SPLIT_TASK}")
else: print("\033[93m\033[1m" + f"\nJoining to help the objective" + "\033[0m\033[0m")

# Results storage using local ChromaDB
class DefaultResultsStorage:
    def __init__(self):
        logging.getLogger('chromadb').setLevel(logging.ERROR)
        # Create Chroma collection
        chroma_persist_dir = "chroma"
        chroma_client = chromadb.Client(
            settings=chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=chroma_persist_dir,
            )
        )

        metric = "cosine"
        embedding_function = InstructorEmbeddingFunction(model_name='hkunlp/instructor-base', device='cuda')
        self.collection = chroma_client.get_or_create_collection(
            name=RESULTS_STORE_NAME,
            metadata={"hnsw:space": metric},
            embedding_function=embedding_function,
        )

    def add(self, task: Dict, result: Dict, result_id: str, vector: List):        
        embeddings = self.collection._embedding_function([vector])

        if (len(self.collection.get(ids=[result_id], include=[])["ids"]) > 0):  # Check if the result already exists
            self.collection.update(
                ids=result_id,
                embeddings=embeddings,
                documents=vector,
                metadatas={"task": task["task_name"], "result": result},
            )
        else:
            self.collection.add(
                ids=result_id,
                embeddings=embeddings,
                documents=vector,
                metadatas={"task": task["task_name"], "result": result},
            )

    def query(self, query: str, top_results_num: int) -> List[dict]:
        count: int = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_texts=query,
            n_results=min(top_results_num, count),
            include=["metadatas"]
        )        
        tasks = []
        count = len(results["ids"][0])
        for i in range(count):            
            resultidstr = results["ids"][0][i]            
            id = int(resultidstr[7:])
            item = results["metadatas"][0][i]            
            task = {'task_id': id, 'task_name': item["task"]}
            tasks.append(task)            
        return tasks
   

# Initialize results storage
results_storage = DefaultResultsStorage()

# Task storage supporting only a single instance of BabyAGI
class SingleTaskListStorage:
    def __init__(self):
        self.tasks = deque([])
        self.task_id_counter = 0

    def append(self, task: Dict):
        self.tasks.append(task)

    def replace(self, tasks: List[Dict]):
        self.tasks = deque(tasks)

    def popleft(self):
        return self.tasks.popleft()

    def is_empty(self):
        return False if self.tasks else True

    def next_task_id(self):
        self.task_id_counter += 1
        return self.task_id_counter

    def get_task_names(self):
        return [t["task_name"] for t in self.tasks]


# Initialize tasks storage
tasks_storage = SingleTaskListStorage()

def ooba_call(prompt: str, history: list[dict] = []) -> str:
    URI=f'{OOBA_API_HOST}:{OOBA_API_PORT}/v1/completions'
    
    request = {
        'prompt': prompt[:CTX_MAX],
        'do_sample': True,
        'temperature': TEMPERATURE,
        'top_p': 0.1,
        'typical_p': 1,
        'repetition_penalty': 1.18,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': [],
        "max_tokens": MAX_NEW_TOKENS
    }

    # TODO: Maybe use the /chat/completions instead to use preset and character?
    """request = {
        "messages": history,
        'preset': 'simple-1',
        'temperature': TEMPERATURE,
        "max_tokens": MAX_NEW_TOKENS,
        "character": "Assistant",
        'top_k': 40,
        "mode": "instruct",
        "frequency_penalty": 1.18
    }"""
    
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(URI, headers=headers, json=request)
    except:
        print('Something went wrong accessing api, is the server running and API enabled?')
        return

    if response.status_code == 200:
        return response.json()['choices'][0]['text']#['message']['content']
    else:
        print("Something went wrong accessing api")

def ollama_call(prompt: str) -> str:
    URI=f'{OLLAMA_API_HOST}:{OLLAMA_API_PORT}/api/generate'
    request = {
        'prompt': prompt[:CTX_MAX],
        'stream': False,
        'model': OLLAMA_MODEL,
        'options':{
            'temperature': TEMPERATURE,
            'top_p': 0.1,
            'typical_p': 1,
            'repeat_penalty': 1.18,
            'top_k': 40,
            'frequency_penalty': 1,
            'num_ctx': CTX_MAX,
            'num_predict': MAX_NEW_TOKENS
        }
    }

    try:
        response = requests.post(URI, json=request)
    except:
        print('Something went wrong accessing api, is Ollama running?')
        return
    
    if response.status_code == 200:
        return response.json()['response']
    else:
        print(f"Something went wrong accessing api. Error: {response.json()['error']}")


def strip_numbered_list(nl: List[str]) -> List[str]:
    result_list = []
    filter_chars = ['#', '(', ')', '[', ']', '.', ':', ' ']

    for line in nl:
        line = line.strip()
        if len(line) > 0:
            parts = line.split(" ", 1)
            if len(parts) == 2:
                left_part = ''.join(x for x in parts[0] if not x in filter_chars)
                if left_part.isnumeric():
                    result_list.append(parts[1].strip())
                else:
                    result_list.append(line)
            else:
                result_list.append(line)

    # filter result_list
    result_list = [line for line in result_list if len(line) > 3]
    
    # remove duplicates
    result_list = list(set(result_list))
    return result_list

def fix_prompt(prompt: str) -> str:
    lines = prompt.split("\n") if "\n" in prompt else [prompt]    
    return "\n".join([line.strip() for line in lines])

def task_creation_agent(
    objective: str, result: Dict, task_description: str, task_list: List[str]
) -> tuple[list[dict], bool]:    
    """prompt = f
    Your objective is: {objective}\n
    You have already completed these tasks, do not repeat them: {task_list if task_list else "None"}\n
    The last task you completed has the result: {result["data"]}\n
    Based on this result and the tasks already completed, please create a list of remaining essential tasks to complete for the purpose of attaining your objective.\n
    Respond with the task list as a numbered list. Don't say anything else. No unnecessary steps. \nResponse:"""
    completed = False
    prompt = f"""
    Your objective is: {objective}\n
    You have already completed these tasks, do not repeat them: {task_list}\n
    The last task you completed has the result: {result["data"]}\n
    Based on the tasks already completed and this result, please create a list of remaining essential tasks to complete for the purpose of attaining your objective.\n
    If you believe there are more tasks that need completion, respond with the task list as a numbered list.\n
    """
    if len(task_list) > 0:
        prompt += "Or If you believe you have completed all necessary tasks to achieve the objective, respond with 'Objective completed.' and nothing else. Never respond with both a list and with 'Objective completed'. \n"
    prompt += """
    Avoid unnecessary steps.\n
    Response:"""
    
    prompt = fix_prompt(prompt)

    if USE_OLLAMA:
        response = ollama_call(prompt)
    else:
        response = ooba_call(prompt)
    pos = response.find("1")
    if (pos > 0):
        response = response[pos - 1:]
    
    if 'Objective completed'.lower() in response.lower():
        response = response.replace('Objective completed.', '')
        #print(f"Objective complete with reasoning: \n{response}")
        completed = True

    if response == '':
        print("\n*** Empty Response from task_creation_agent***")
        new_tasks_list = result["data"].split("\n") if len(result) > 0 else [response]
    else:
        new_tasks = response.split("\n") if "\n" in response else [response]
        new_tasks_list = strip_numbered_list(new_tasks)
        
    return [{"task_name": task_name} for task_name in (t for t in new_tasks_list if not t == '')], completed


def prioritization_agent():
    task_names = tasks_storage.get_task_names()
    next_task_id = tasks_storage.next_task_id()    

    prompt = f"""
        Your ultimate objective is: {OBJECTIVE}

        To achieve this objective, you have the following tasks to complete: {task_names}

        Take this task list and perform the following actions:

        1. Reorder the task list so that tasks that need to be completed before other tasks come first.
        2. Summarize each item in the reordered task list.
        3. Consolidate the reordered, summarized task list so that there are no repeated tasks and so that there are at most {MAX_TASKS} tasks in the list.
        4. Respond with the reordered, summarized, consolidated task list as a numbered list.

        Please provide the revised task list as a numbered list. Do not include any additional information.

        Response:
    """
    
    prompt = fix_prompt(prompt)

    if USE_OLLAMA:
        response = ollama_call(prompt)
    else:
        response = ooba_call(prompt)
    pos = response.find("1")
    if (pos > 0):
        response = response[pos - 1:]

    new_tasks = response.split("\n") if "\n" in response else [response]
    new_tasks = strip_numbered_list(new_tasks)
    new_tasks_list = []
    i = 0
    for task_string in new_tasks:        
        new_tasks_list.append({"task_id": i + next_task_id, "task_name": task_string})
        i += 1
    
    if len(new_tasks_list) > 0:
        tasks_storage.replace(new_tasks_list)


# Execute a task based on the objective and five previous tasks
def execution_agent(objective: str, task: str, history: list[dict] = []) -> str:
    """
    Executes a task based on the given objective and previous context.

    Args:
        objective (str): The objective or goal for the AI to perform the task.
        task (str): The task to be executed by the AI.

    Returns:
        str: The response generated by the AI for the given task.

    """
    
    context = context_agent(query=objective, top_results_num=MAX_TASKS)

    context_list = [t['task_name'] for t in context if t['task_name'] != OBJECTIVE_SPLIT_TASK]
    #context_list = [t['task_name'] for t in context]

    # remove duplicates
    context_list = list(set(context_list))    

    if VERBOSE and len(context_list) > 0:
        print("\n*******RELEVANT CONTEXT******\n")
        print(context_list)

    if task == OBJECTIVE_SPLIT_TASK:
        prompt = f"""
        Your objective is: {objective}\n
        Please complete the following task: {task}\n
        Do not ask any clarifying questions.\nResponse:"""
        
    else:  
        prompt = f"""
            Your objective: {OBJECTIVE}

            Your current task is: {task}

            {"You have already completed the following tasks, take them into account as you complete the task but do not repeat them: " + str(context_list) if len(context_list)>0 else ""}

            1. Determine if the task involves fact checking or information gathering.
            2. If the task involves fact-checking, information gathering or any necessary actions, such as calculations or getting the current date, then respond with a newly generated search string for the task and place the string between {ooba_web.SEARCH_START} and {ooba_web.SEARCH_END}.
            3. If the task does not involve fact checking or information gathering, respond with the completed task.

            Do not ask any clarifying questions. Do not clarify your reasoning.

            Response:
        """
    prompt = fix_prompt(prompt)
    if USE_OLLAMA:
        result = ollama_call(prompt)
    else:
        result = ooba_call(prompt, history)

    if result and task == OBJECTIVE_SPLIT_TASK:
        pos = result.find("1")
        if (pos > 0):
            result = result[pos - 1:]
    return result

def search_agent(task, query: str, search_results: list) -> str:
    """
    TODO
    """

    prompt = (
        f'Your current task is: "{task}".\n ## Search Results for: "{query}"\n'
        "Please review the following search results carefully and select the index of the single webpage that best addresses the task."
        "\n\n"
    ) + "\n\n".join(
        f"Index number: {r+1} \"{search_results[r]['title']}\"\n"
        f"**URL:** {search_results[r]['url']}  \n"
        f"Exerpt: {search_results[r]['exerpt']}\n"
        for r in range(len(search_results))
    ) + "\n\nRespond with the index of the single webpage that best addresses the task:"

    prompt = ooba_web.safe_google_results(prompt)
    prompt = fix_prompt(prompt)

    if USE_OLLAMA:
        return ollama_call(prompt)
    return ooba_call(prompt)

def search_extract_agent(task: str, search_query: str, block_text: str) -> str:
    max_length = 400 # TODO: find a good max value
    prompt = f"""
        Objective: {OBJECTIVE}

        Task: {task}

        Search Query: {search_query}

        Block Text:
        {block_text[:max_length]}

        Extract the relevant information from the block text to complete the task. Consider the following guidelines:
        1. Identify key points or details related to the task.
        2. Summarize the information in a concise manner.
        3. Ensure that the extracted information directly addresses the objective and task provided.

        Provide your response based on the extracted information:

        Response:
    """
    prompt = fix_prompt(prompt)
    if USE_OLLAMA:
        return ollama_call(prompt)
    return ooba_call(prompt)

def updated_result_agent(task: str, initial_answer: str, new_search_results: list[tuple]) -> str:

    prompt = f"""
        Overall Objective: {OBJECTIVE}

        Task: {task}

        Initial answer to the task {initial_answer}

        New Search Results:
    """

    for search_query, search_result in new_search_results:
        prompt += f"\n\nSearch Query: {search_query}\nSearch Result:\n{search_result}"

    prompt += f"""

    Instructions:
    1. Review the initial search results and the new information obtained from the additional searches.
    2. Consider how the new data alters or enhances the understanding of the task.
    3. Based on the updated information, provide a revised answer to the task.
    4. Ensure that your response is relevant to the overall objective and accurately addresses the task.

    Provide your response based on the updated information:

    Response:
    """
    prompt = fix_prompt(prompt)
    if USE_OLLAMA:
        return ollama_call(prompt)
    return ooba_call(prompt)

# Get the top n completed tasks for the objective
def context_agent(query: str, top_results_num: int):
    """
    Retrieves context for a given query from an index of tasks.

    Args:
        query (str): The query or objective for retrieving context.
        top_results_num (int): The number of top results to retrieve.

    Returns:
        list: A list of tasks as context for the given query, sorted by relevance.

    """
    results = results_storage.query(query=query, top_results_num=top_results_num)
    #print("\n***** RESULTS *****")
    #print(results)
    return results

# Add the initial task if starting new objective
if not JOIN_EXISTING_OBJECTIVE:
    initial_task = {
        "task_id": tasks_storage.next_task_id(),
        "task_name": OBJECTIVE_SPLIT_TASK
    }
    tasks_storage.append(initial_task)

def main():
    while True:
        crawler = ooba_web.Crawler()
        was_search_used = False
        # As long as there are tasks in the storage...
        if not tasks_storage.is_empty():
            # Print the task list
            print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
            for t in tasks_storage.get_task_names():
                print(" • "+t)

            # Step 1: Pull the first incomplete task
            task = tasks_storage.popleft()
            print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
            print(task['task_name'])

            result = execution_agent(OBJECTIVE, task["task_name"])
            if not result:
                print('Error occoured. Check if model is loaded.')
                break
            
            search_required = ooba_web.check_for_search(result)
            if search_required:
                search_strings = ooba_web.get_search_strings(result)
                complete_search_results = []
                for search_string in search_strings:
                    search_results = crawler.ddg_search(search_string[0], num_results=4)
                    if len(search_results) > 0:
                        result_search = search_agent(task["task_name"], search_string[0], search_results)
                        ind_str = re.findall(r'\d+', result_search)
                        
                        if len(ind_str) == 0:
                            continue
                        ind = int(ind_str[0]) - 1
                        if len(search_results) < ind:
                            continue

                        selected_search_page = search_results[ind]
                        print(f"Fetching data About: {search_string[0]}.\nURL: {selected_search_page['url']}.")
                        crawl_data = crawler.crawl(selected_search_page['url'])
                        if not crawl_data:
                            continue
                        result_search_agent = search_extract_agent(task['task_name'], search_string[0], crawl_data)
                        complete_search_results.append((search_string[0], result_search_agent))

                result = updated_result_agent(task["task_name"], result, complete_search_results)

            print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
            print(result)

            # Step 2: Enrich result and store in the results storage
            # This is where you should enrich the result if needed
            enriched_result = {
                "data": result
            }
            # extract the actual result from the dictionary
            # since we don't do enrichment currently
            vector = enriched_result["data"]  

            result_id = f"result_{task['task_id']}"
            results_storage.add(task, result, result_id, vector)

            # Step 3: Create new tasks and reprioritize task list
            # only the main instance in cooperative mode does that
            new_tasks, completed = task_creation_agent(
                OBJECTIVE,
                enriched_result,
                task["task_name"],
                tasks_storage.get_task_names(),
            )

            if completed:
                if was_search_used:
                    print('\nThe model did not search for any information online.')
                break

            for new_task in new_tasks:
                if not new_task['task_name'] == '':
                    new_task.update({"task_id": tasks_storage.next_task_id()})
                    tasks_storage.append(new_task)

            if not JOIN_EXISTING_OBJECTIVE: prioritization_agent()

            # Sleep a bit before checking the task list again
            time.sleep(1)

        else:
            print ("Ready, no more tasks.")

if __name__ == "__main__":
    main()
