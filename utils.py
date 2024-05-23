import os
import copy
import json
import time
import yaml
import random
import requests

from typing import Optional
from glob import glob

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0613-verbose",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
)


temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}


def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(
        endpoint_list
    )[0]
    return api_dict


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None):
    import openai
    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            # print(messages)
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
                )
            output = completion.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except KeyError:
            print(type(e), e)
            break
    
    return output


def chat_completion_openai_azure(model, messages, temperature, max_tokens, api_dict=None):
    import openai
    from openai import AzureOpenAI

    api_base = api_dict["api_base"]
    client = AzureOpenAI(
        azure_endpoint = api_base,
        api_key= api_dict["api_key"],
        api_version=api_dict["api_version"],
        timeout=240,
        max_retries=2
    )

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=42,
            )
            output = response.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(type(e), e)
            break

    return output


def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None):
    import anthropic

    if api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            # print(sys_msg)
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output


def chat_completion_mistral(model, messages, temperature, max_tokens):
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.exceptions import MistralException

    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)

    prompts = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            chat_response = client.chat(
                model=model,
                messages=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = chat_response.choices[0].message.content
            break
        except MistralException as e:
            print(type(e), e)
            break

    return output


def chat_completion_gemini(model, messages, temperature, max_tokens):
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]

    # Set up the model
    generation_config = {
        "temperature": temperature,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": max_tokens,
    }

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            gemini = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                safety_settings=safety_settings)

            convo = gemini.start_chat(history=[])
            
            convo.send_message(messages)
            output = convo.last.text
            break
        except genai.types.generation_types.StopCandidateException as e:
            print(type(e), e)
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    
    return output


def chat_completion_cohere(model, messages, temperature, max_tokens):
    import cohere

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    assert len(messages) > 0

    template_map = {"system":"SYSTEM",
                    "assistant":"CHATBOT",
                    "user":"USER"}

    assert messages[-1]["role"] == "user"
    prompt = messages[-1]["content"]

    if len(messages) > 1:
        history = []
        for message in messages[:-1]:
            history.append({"role":template_map[message["role"]], "message":message["content"]})
    else:
        history = None

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = co.chat(
                message=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=history,
            )
            output = response.text
            break
        except cohere.core.api_error.ApiError as e:
            print(type(e), e)
            raise
        except Exception as e:
            print(type(e), e)
            break
    
    return output


def get_response_batch(prompts, url, max_tokens, temperature, api_key, api_args, retries_left=5):
    import requests
    headers = {"Authorization": api_key, "Content-Type": "application/json"}

    data = {
        "prompt": prompts, 
        # "messages": prompts,
        "temperature": temperature,
        "max_tokens": 1024,
        "use_raw_prompt": True,
        "stop": ["<|im_end|>"],
        "model": '',
        **api_args
    }
    
    # Remove things from payload
    for k,v in api_args.items():
        if v is None:
            data.pop(k)
    
    print ("temperature is: ", temperature)
    
    response = requests.post(url, headers=headers, json=data, timeout=360)

    if response.status_code == 400 and "Please reduce the length of your prompt." in response.text:
            return None
    elif  response.status_code != 200:
        print(response.status_code)
        print(response.text)
        if retries_left > 0:
            print("Retrying...")
            # sleep for longer each retry
            time.sleep(5 * (6 - retries_left))
            return get_response_batch(prompts, url, max_tokens, temperature=temperature, api_key=api_key, api_args=api_args, retries_left=retries_left-1)
        else:
            raise Exception("Too many retries")
    else:
        response = response.json()
        
        # need to trim the leading space on all choices
        responses = []
        finish_reasons = []
        for i, choice in enumerate(response["choices"]):
            if "text" in choice:
                responses.append(choice["text"].strip())
                finish_reasons.append(choice["finish_reason"])
            else: # assuming chat
                responses.append(choice["message"]["content"].strip())
                finish_reasons.append(choice["finish_reason"])
        return responses# , finish_reasons, response['usage']['prompt_tokens'], response['usage']['completion_tokens']


def pairwise_reward_model_inf(url, input_ids, retries_left=5):
    custom_input = [{
        'input_ids': input_ids,
    }]

    headers = {"Authorization": os.environ['MOSAICML_API_KEY'],
               "Content-Type": "application/json"}

    request = {'custom_input': custom_input, 'prompt': ''}
    inf_data = json.dumps(request)

    response = requests.post(f'{url}',
                                headers=headers,
                                data=inf_data,
                                timeout=360)

    if response.status_code == 400 and 'Please reduce the length of your prompt.' in response.text:
        return None
    elif response.status_code != 200:
        if retries_left > 0:
            print("Retrying...", response.status_code)
            # sleep for longer each retry
            time.sleep(5 * (6 - retries_left))
            return pairwise_reward_model_inf(url, input_ids, retries_left=retries_left - 1)
        else:
            raise Exception('Too many retries')
    else:
        response = response.json()
        # print("REWARD CALL", response)

        final_rewards = []
        for choice in response['choices']:
            final_reward = choice['metadata']['rewards'][-1]
            final_rewards.append(final_reward)
        return final_rewards


def format_messages_for_reward(messages, responses, tokenizer):
    messages.append({'role': 'assistant', 'content': responses})
    rm_input = tokenizer.apply_chat_template(messages, tokenize = True) + [tokenizer.eos_token_id]
    messages.pop()
    return rm_input


def ping_db_with_messages(messages, base_url, temperature, top_p=1):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ['DATABRICKS_TOKEN'],
        base_url=base_url
    )

    print (messages)

    chat_completion = client.chat.completions.create(
        messages=messages, # Chat formatted messages
        model='databricks-dbrx-instruct',
        temperature=temperature,
        top_p=top_p,
    )
    response = chat_completion.choices[0].message.content
    return response


def db_inference_deployment(model, tokenizer, messages, temperature, max_tokens, api_key, api_args={}, api_dict=None, reward_model_addr=None, num_rm_samples=1):
    from transformers import AutoTokenizer
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    orig_len = len(messages)

    chosen_reward_score = None
    if reward_model_addr is not None and num_rm_samples > 1:
        samples = []

        for i in range(num_rm_samples):
            print ("i is: ", i, "temperature is: ", temperature)
            assert len(messages) == orig_len

            while True:
                responses = get_response_batch(prompt, model, max_tokens, api_key=api_key, api_args=api_args, temperature = temperature)[0]

                rm_messages = messages
                rm_messages.append({'role': 'assistant', 'content': responses})
                assert len(rm_messages) == orig_len + 1
                rm_input = tokenizer.apply_chat_template(rm_messages, tokenize = True) + [tokenizer.eos_token_id]

                # resample if we go over the RM limit
                # TODO: don't hard code
                if len(rm_input) > 4096:
                    print ("resampling because rm input is: ", len(rm_input))
                    # Remove the last appended message since it's too long for the reward mdoel
                    rm_messages.pop()
                    continue

                reward_score = pairwise_reward_model_inf(reward_model_addr, rm_input)[0]
                break

            samples.append((responses, reward_score))

            # remove the last appended message
            rm_messages.pop()

        samples = sorted(samples, key = lambda x: x[1])
        output = samples[-1][0]
        chosen_reward_score = samples[-1][1]

    else:
        if 'serving-endpoints' in model:
            responses = ping_db_with_messages(messages, model, temperature)

        else:
            responses = get_response_batch(prompt, model, max_tokens, api_key=api_key, api_args=api_args, temperature = temperature)[0]

            if 'Confidence: ' in responses:
                responses = responses[:responses.find('Confidence: ')]

        if reward_model_addr is not None:
            rm_messages = format_messages_for_reward (messages, responses, tokenizer)
            chosen_reward_score = pairwise_reward_model_inf(reward_model_addr, rm_messages)[0]
            print ("reward score is: ", chosen_reward_score)

        output = responses

    # print (output)
    return output, chosen_reward_score


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])
