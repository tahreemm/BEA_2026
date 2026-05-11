import yaml
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio
import random
import time
from groq import Groq
# Load environment variables
load_dotenv()
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
model = "llama-3.3-70b-versatile"

async def _make_student_call_async(givens, conclusion, intermediates):
    # Load prompt from YAML file
    prompt_file = Path('/Users/tahreemyasir/Documents/prelims/DT_hint-1/dt_code/prompts_i+2/student_prompts.yaml')
    
    with open(prompt_file, 'r') as f:
        prompts = yaml.safe_load(f)
    
    # Get the student prompt template
    prompt_template = prompts['student_prompt']
    
    # System prompt: Role, Problem_format, Constraints, Response_Format
    system_prompt = f"""{prompt_template['role']}

    {prompt_template['Step_by_Step_Instructions']}

    {prompt_template['Constraints']}

    {prompt_template['Response_Format']}
    """
    
    # User prompt: variables
    user_prompt = f"""

    GIVENS: {givens}
    INTERMEDIATE_STEPS: {intermediates}
    CONCLUSION: {conclusion}"""

    try:
        max_retries = 10
        loop = asyncio.get_event_loop()
        response_text = None
        
        for attempt in range(max_retries):
            # Initial wait before first attempt, or exponential backoff for retries
            if attempt == 0:
                initial_wait = 40  # 20 seconds initial wait
                await asyncio.sleep(initial_wait)
            else:
                wait_time = (2 ** attempt) * 50 + random.uniform(0, 10)
                print(f"Empty student response, retrying {attempt}/{max_retries} after {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
            start = time.perf_counter()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    top_p=0.9,
                )
            )
            end = time.perf_counter()
            print(f"Time taken: {end - start} seconds")

            # Extract text
            response_text = response.choices[0].message.content

            # Token usage
            completion_tokens = response.usage.completion_tokens
            prompt_tokens = response.usage.prompt_tokens
            total_tokens = completion_tokens + prompt_tokens
            
            if response_text and response_text.strip() and len(response_text.strip()) >= 10:
                return response_text, completion_tokens, total_tokens, end - start
        
        print(f"Failed to get valid response after {max_retries} attempts")
        return response_text or "Error: No response received", None, None, None
    except Exception as e:
        print(f"Error in make_student_call: {e}")
        return f"Error: {str(e)}", None, None, None

def make_student_call(givens, conclusion, intermediates):
    """Synchronous wrapper for async student call"""
    return asyncio.run(_make_student_call_async(givens, conclusion, intermediates))



async def _make_teacher_only_call_async(givens, conclusion, intermediates, correct_step, student_response):
    # Load prompt from YAML file
    prompt_file = Path('/Users/tahreemyasir/Documents/prelims/DT_hint-1/dt_code/prompts_i+2/teacher_prompt.yaml')
    
    with open(prompt_file, 'r') as f:
        prompts = yaml.safe_load(f)
    
    # Get the teacher prompt template
    prompt_template = prompts['teacher_only_prompt']
    
    # System prompt: Role, Problem_Format, Step_by_Step_Instructions, Constraints, Response_Format
    system_prompt = f"""{prompt_template['role']}

    {prompt_template['Step_by_Step_Instructions']}

    {prompt_template['Constraints']}

    {prompt_template['Response_Format']}"""
    
    
    # User prompt: variables
    user_prompt = f"""
    GIVENS: {givens}
    INTERMEDIATE_STEPS: {intermediates}
    CONCLUSION: {conclusion}
    CORRECT_STEP: {correct_step}
    STUDENT_RESPONSE: {student_response}"""

    try:
        max_retries = 10
        loop = asyncio.get_event_loop()
        response_text = None
        
        for attempt in range(max_retries):
            # Initial wait before first attempt, or exponential backoff for retries
            if attempt == 0:
                initial_wait = 40  # 20 seconds initial wait
                await asyncio.sleep(initial_wait)
            else:
                wait_time = (2 ** attempt) * 50 + random.uniform(0, 10)
                print(f"Empty teacher response, retrying {attempt}/{max_retries} after {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
            start = time.perf_counter()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    top_p=0.9,
                )

            )
            end = time.perf_counter()
            print(f"Time taken: {end - start} seconds")

            # Extract text
            response_text = response.choices[0].message.content
            # Token usage
            completion_tokens = response.usage.completion_tokens
            prompt_tokens = response.usage.prompt_tokens
            total_tokens = completion_tokens + prompt_tokens
                
            if response_text and response_text.strip() and len(response_text.strip()) >= 10:
                return response_text, completion_tokens, total_tokens, end - start
        
        print(f"Failed to get valid response after {max_retries} attempts")
        return response_text or "Error: No response received", None, None, None
    except Exception as e:
        print(f"Error in make_teacher_only_call: {e}")
        return f"Error: {str(e)}", None, None, None

def make_teacher_only_call(givens, conclusion, intermediates, correct_step, student_response):
    """Synchronous wrapper for async teacher call"""
    return asyncio.run(_make_teacher_only_call_async(givens, conclusion, intermediates, correct_step, student_response))


async def _make_judge_only_call_async(givens, conclusion, intermediates, knowledge_graph_steps, student_response):
    # Load prompt from YAML file
    prompt_file = Path('/Users/tahreemyasir/Documents/prelims/DT_hint-1/dt_code/prompts_i+2/judge_prompt.yaml')
    
    with open(prompt_file, 'r') as f:
        prompts = yaml.safe_load(f)
    
    # Get the judge prompt template
    prompt_template = prompts['judge_only_prompt']
    
    # System prompt: Role, Instructions, Example, Constraints, Response_Format
    system_prompt = f"""{prompt_template['role']}

    {prompt_template['Step_by_Step_Instructions']}

    {prompt_template['Constraints']}

    {prompt_template['Response_Format']}"""
    
    # User prompt: variables
    user_prompt = f"""
    GIVENS: {givens}
    INTERMEDIATE_STEPS: {intermediates}
    CONCLUSION: {conclusion}
    STUDENT_RESPONSE: {student_response}
    KNOWLEDGE_GRAPH_STEPS: {knowledge_graph_steps}"""

    try:
        max_retries = 10
        loop = asyncio.get_event_loop()
        response_text = None
        
        for attempt in range(max_retries):
            # Initial wait before first attempt, or exponential backoff for retries
            if attempt == 0:
                initial_wait = 40  # 20 seconds initial wait
                await asyncio.sleep(initial_wait)
            else:
                wait_time = (2 ** attempt) * 50 + random.uniform(0, 10)
                print(f"Empty judge response, retrying {attempt}/{max_retries} after {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
            start = time.perf_counter()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    top_p=0.9,
                )
            )
            end = time.perf_counter()
            print(f"Time taken: {end - start} seconds")

            # Extract text
            response_text = response.choices[0].message.content

            # Token usage
            completion_tokens = response.usage.completion_tokens
            prompt_tokens = response.usage.prompt_tokens
            total_tokens = completion_tokens + prompt_tokens
            
            if response_text and response_text.strip() and len(response_text.strip()) >= 10:
                return response_text, completion_tokens, total_tokens, end - start
        
        print(f"Failed to get valid response after {max_retries} attempts")
        return response_text or "Error: No response received", None, None, None
    except Exception as e:
        print(f"Error in make_judge_only_call: {e}")
        return f"Error: {str(e)}", None, None, None

def make_judge_only_call(givens, conclusion, intermediates, knowledge_graph_steps, student_response):
    """Synchronous wrapper for async judge call"""
    return asyncio.run(_make_judge_only_call_async(givens, conclusion, intermediates, knowledge_graph_steps, student_response))

async def _make_judge_verifier_call_async(givens, conclusion, intermediates, student_response, teacher_response, knowledge_base_steps):
    # Load prompt from YAML file
    prompt_file = Path('/Users/tahreemyasir/Documents/prelims/DT_hint-1/dt_code/prompts_i+2/judge_prompt.yaml')
    
    with open(prompt_file, 'r') as f:
        prompts = yaml.safe_load(f)
    
    # Get the judge prompt template
    prompt_template = prompts['judge_verifier_prompt']
    
    # System prompt: Role, Instructions, Example, Constraints, Response_Format
    system_prompt = f"""{prompt_template['role']}

    {prompt_template['Step_by_Step_Instructions']}
    {prompt_template['Constraints']}
    {prompt_template['Response_Format']}"""
    
    # User prompt: variables
    user_prompt = f"""
    GIVENS: {givens}
    INTERMEDIATE_STEPS: {intermediates}
    CONCLUSION: {conclusion}
    STUDENT_RESPONSE: {student_response}
    TEACHER_RESPONSE: {teacher_response}
    KNOWLEDGE_BASE_STEPS: {knowledge_base_steps}"""

    try:
        max_retries = 10
        loop = asyncio.get_event_loop()
        response_text = None
        
        for attempt in range(max_retries):
            # Initial wait before first attempt, or exponential backoff for retries
            if attempt == 0:
                initial_wait = 40  # 20 seconds initial wait
                await asyncio.sleep(initial_wait)
            else:
                wait_time = (2 ** attempt) * 50 + random.uniform(0, 10)
                print(f"Empty judge response, retrying {attempt}/{max_retries} after {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
            start = time.perf_counter()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    top_p=0.9,
                )
            )

            end = time.perf_counter()
            print(f"Time taken: {end - start} seconds")

            # Extract text
            response_text = response.choices[0].message.content

            # Token usage
            completion_tokens = response.usage.completion_tokens
            prompt_tokens = response.usage.prompt_tokens
            total_tokens = completion_tokens + prompt_tokens
            
            if response_text and response_text.strip() and len(response_text.strip()) >= 10:
                return response_text, completion_tokens, total_tokens, end - start
        
        print(f"Failed to get valid response after {max_retries} attempts")
        return response_text or "Error: No response received", None, None, None
    except Exception as e:
        print(f"Error in make_judge_only_call: {e}")
        return f"Error: {str(e)}", None, None, None

def make_judge_verifier_call(givens, conclusion, intermediates, student_response, teacher_response, knowledge_base_steps):
    """Synchronous wrapper for async judge call"""
    return asyncio.run(_make_judge_verifier_call_async(givens, conclusion, intermediates, student_response, teacher_response, knowledge_base_steps))


async def _make_student_update_call_async(givens, conclusion, intermediates, previous_student_response, feedback):
    # Load prompt from YAML file
    prompt_file = Path('/Users/tahreemyasir/Documents/prelims/DT_hint-1/dt_code/prompts_i+2/student_prompts.yaml')
    
    with open(prompt_file, 'r') as f:
        prompts = yaml.safe_load(f)
    
    # Get the student update prompt template
    prompt_template = prompts['student_update_prompt']
    
    # System prompt: Role, Instructions, Example, Constraints, Response_Format
    system_prompt = f"""{prompt_template['role']}

    {prompt_template['Step_by_Step_Instructions']}

    {prompt_template['Response_Format']}"""
    
    # User prompt: variables
    user_prompt = f"""
    GIVENS: {givens}
    INTERMEDIATE_STEPS: {intermediates}
    CONCLUSION: {conclusion}
    PREVIOUS_STUDENT_RESPONSE: {previous_student_response}
    FEEDBACK: {feedback}"""

    try:
        max_retries = 10
        loop = asyncio.get_event_loop()
        response_text = None
        
        for attempt in range(max_retries):
            # Initial wait before first attempt, or exponential backoff for retries
            if attempt == 0:
                initial_wait = 40  # 40 seconds initial wait
                await asyncio.sleep(initial_wait)
            else:
                wait_time = (2 ** attempt) * 50 + random.uniform(0, 10)
                print(f"Empty student update response, retrying {attempt}/{max_retries} after {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
            start = time.perf_counter()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    top_p=0.9,
                )
            )

            end = time.perf_counter()
            print(f"Time taken: {end - start} seconds")

            # Extract text
            response_text = response.choices[0].message.content

            # Token usage
            completion_tokens = response.usage.completion_tokens
            prompt_tokens = response.usage.prompt_tokens
            total_tokens = completion_tokens + prompt_tokens
            
            if response_text and response_text.strip() and len(response_text.strip()) >= 10:
                return response_text, completion_tokens, total_tokens, end - start
        
        print(f"Failed to get valid response after {max_retries} attempts")
        return response_text or "Error: No response received", None, None, None
    except Exception as e:
        print(f"Error in make_student_update_call: {e}")
        return f"Error: {str(e)}", None, None, None

def make_student_update_call(givens, conclusion, intermediates, previous_student_response, feedback):
    """Synchronous wrapper for async student update call"""
    return asyncio.run(_make_student_update_call_async(givens, conclusion, intermediates, previous_student_response, feedback))

    

