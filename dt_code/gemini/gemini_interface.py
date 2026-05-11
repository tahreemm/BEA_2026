import yaml
from pathlib import Path
import google.generativeai as genai
import os
from dotenv import load_dotenv
import asyncio
import random
import time

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-pro')

async def _make_student_call_async(givens, conclusion, intermediates):
    """Make student call using Gemini 2.5 Flash"""
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
        max_retries = 3
        loop = asyncio.get_event_loop()
        response_text = None
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        for attempt in range(max_retries):
            # Initial wait before first attempt, or exponential backoff for retries
            if attempt == 0:
                initial_wait = 30  # 20 seconds initial wait
                await asyncio.sleep(initial_wait)
            else:
                wait_time = (2 ** attempt) * 20 + random.uniform(0, 10)
                print(f"Empty student response, retrying {attempt}/{max_retries} after {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
                
            start = time.perf_counter()
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            def _generate():
                return model.generate_content(full_prompt)
            end = time.perf_counter()
            print(f"Time taken: {end - start} seconds")
            response = await loop.run_in_executor(None, _generate)
            response_text = response.text
            candidates_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count
            
            
            if response_text and response_text.strip() and len(response_text.strip()) >= 10:
                return response_text, candidates_tokens, total_tokens, end - start
        
        print(f"Failed to get valid response after {max_retries} attempts")
        return response_text or "Error: No response received", None, None, None
    except Exception as e:
        print(f"Error in make_student_call: {e}")
        return f"Error: {str(e)}", None, None, None

def make_student_call(givens, conclusion, intermediates):
    """Synchronous wrapper for async student call"""
    return asyncio.run(_make_student_call_async(givens, conclusion, intermediates))

async def _make_teacher_only_call_async(givens, conclusion, intermediates, correct_step, student_response):
    """Make teacher only call using Gemini 2.5 Flash"""
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
        max_retries = 3
        loop = asyncio.get_event_loop()
        response_text = None
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        for attempt in range(max_retries):
            # Initial wait before first attempt, or exponential backoff for retries
            if attempt == 0:
                initial_wait = 30  # 20 seconds initial wait
                await asyncio.sleep(initial_wait)
            else:
                wait_time = (2 ** attempt) * 20 + random.uniform(0, 10)
                print(f"Empty teacher response, retrying {attempt}/{max_retries} after {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
            start = time.perf_counter()
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            def _generate():
                return model.generate_content(full_prompt)
            end = time.perf_counter()
            print(f"Time taken: {end - start} seconds")
            response = await loop.run_in_executor(None, _generate)
            response_text = response.text
            candidates_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count
            
            
            if response_text and response_text.strip() and len(response_text.strip()) >= 10:
                return response_text, candidates_tokens, total_tokens, end - start
        
        print(f"Failed to get valid response after {max_retries} attempts")
        return response_text or "Error: No response received", None, None, None
    except Exception as e:
        print(f"Error in make_teacher_only_call: {e}")
        return f"Error: {str(e)}", None, None, None 

def make_teacher_only_call(givens, conclusion, intermediates, correct_step, student_response):
    """Synchronous wrapper for async teacher call"""
    return asyncio.run(_make_teacher_only_call_async(givens, conclusion, intermediates, correct_step, student_response))

async def _make_judge_only_call_async(givens, conclusion, intermediates, student_response, knowledge_graph_steps):
    """Make judge only call using Gemini 2.5 Flash"""
    # Load prompt from YAML file
    prompt_file = Path('/Users/tahreemyasir/Documents/prelims/DT_hint-1/dt_code/prompts_i+2/judge_prompt.yaml')
    
    with open(prompt_file, 'r') as f:
        prompts = yaml.safe_load(f)
    
    # Get the judge prompt template
    prompt_template = prompts['judge_only_prompt']
    
    # System prompt: Role, Step_by_Step_Instructions, Constraints, Response_Format
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
        max_retries = 3
        loop = asyncio.get_event_loop()
        response_text = None
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        for attempt in range(max_retries):
            # Initial wait before first attempt, or exponential backoff for retries
            if attempt == 0:
                initial_wait = 30  # 20 seconds initial wait
                await asyncio.sleep(initial_wait)
            else:
                wait_time = (2 ** attempt) * 20 + random.uniform(0, 10)
                print(f"Empty judge response, retrying {attempt}/{max_retries} after {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
            start = time.perf_counter()
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            def _generate():
                return model.generate_content(full_prompt)
            end = time.perf_counter()
            print(f"Time taken: {end - start} seconds")
            response = await loop.run_in_executor(None, _generate)
            response_text = response.text
            candidates_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count
            
            if response_text and response_text.strip() and len(response_text.strip()) >= 10:
                return response_text, candidates_tokens, total_tokens, end - start
        
        print(f"Failed to get valid response after {max_retries} attempts")
        return response_text or "Error: No response received", None, None, None
    except Exception as e:
        print(f"Error in make_judge_only_call: {e}")
        return f"Error: {str(e)}", None, None, None

def make_judge_only_call(givens, conclusion, intermediates, student_response, knowledge_graph_steps):
    """Synchronous wrapper for async judge only call"""
    return asyncio.run(_make_judge_only_call_async(givens, conclusion, intermediates, student_response, knowledge_graph_steps))


async def _make_judge_verifier_call_async(givens, conclusion, intermediates, student_response, teacher_response, knowledge_graph_steps):
    """Make judge verifier call using Gemini 2.5 Flash"""
    # Load prompt from YAML file
    prompt_file = Path('/Users/tahreemyasir/Documents/prelims/DT_hint-1/dt_code/prompts_i+2/judge_prompt.yaml')
    
    with open(prompt_file, 'r') as f:
        prompts = yaml.safe_load(f)
    
    # Get the judge verifier prompt template
    prompt_template = prompts['judge_verifier_prompt']
    
    # System prompt: Role, Step_by_Step_Instructions, Constraints, Response_Format
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
    KNOWLEDGE_GRAPH_STEPS: {knowledge_graph_steps}"""

    try:
        max_retries = 3
        loop = asyncio.get_event_loop()
        response_text = None
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        for attempt in range(max_retries):
            # Initial wait before first attempt, or exponential backoff for retries
            if attempt == 0:
                initial_wait = 30  # 20 seconds initial wait
                await asyncio.sleep(initial_wait)
            else:
                wait_time = (2 ** attempt) * 20 + random.uniform(0, 10)
                print(f"Empty judge verifier response, retrying {attempt}/{max_retries} after {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
            start = time.perf_counter()
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            def _generate():
                return model.generate_content(full_prompt)
            end = time.perf_counter()
            print(f"Time taken: {end - start} seconds")
            response = await loop.run_in_executor(None, _generate)
            response_text = response.text
            candidates_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count
            
            if response_text and response_text.strip() and len(response_text.strip()) >= 10:
                return response_text, candidates_tokens, total_tokens, end - start
        
        print(f"Failed to get valid response after {max_retries} attempts")
        return response_text or "Error: No response received", None, None, None
    except Exception as e:
        print(f"Error in make_judge_verifier_call: {e}")
        return f"Error: {str(e)}", None, None, None

def make_judge_verifier_call(givens, conclusion, intermediates, student_response, teacher_response, knowledge_graph_steps):
    """Synchronous wrapper for async judge verifier call"""
    return asyncio.run(_make_judge_verifier_call_async(givens, conclusion, intermediates, student_response, teacher_response, knowledge_graph_steps))

async def _make_student_update_call_async(givens, conclusion, intermediates, previous_student_response, feedback):
    """Make student update call using Gemini 2.5 Flash"""
    # Load prompt from YAML file
    prompt_file = Path('/Users/tahreemyasir/Documents/prelims/DT_hint-1/dt_code/prompts_i+2/student_prompts.yaml')
    
    with open(prompt_file, 'r') as f:
        prompts = yaml.safe_load(f)
    
    # Get the student update prompt template
    prompt_template = prompts['student_update_prompt']    
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
    PREVIOUS_STUDENT_RESPONSE: {previous_student_response}
    FEEDBACK: {feedback}"""

    try:
        max_retries = 3
        loop = asyncio.get_event_loop()
        response_text = None
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        for attempt in range(max_retries):
            # Initial wait before first attempt, or exponential backoff for retries
            if attempt == 0:
                initial_wait = 30  # 20 seconds initial wait
                await asyncio.sleep(initial_wait)
            else:
                wait_time = (2 ** attempt) * 20 + random.uniform(0, 10)
                print(f"Empty student update response, retrying {attempt}/{max_retries} after {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
            start = time.perf_counter()
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            def _generate():
                return model.generate_content(full_prompt)
            end = time.perf_counter()
            print(f"Time taken: {end - start} seconds")
            response = await loop.run_in_executor(None, _generate)
            response_text = response.text
            candidates_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count
            if response_text and response_text.strip() and len(response_text.strip()) >= 10:
                return response_text, candidates_tokens, total_tokens, end - start
        
        print(f"Failed to get valid response after {max_retries} attempts")
        return response_text or "Error: No response received", None, None, None
    except Exception as e:
        print(f"Error in make_student_update_call: {e}")
        return f"Error: {str(e)}", None, None, None   

def make_student_update_call(givens, conclusion, intermediates, previous_student_response, feedback):
    """Synchronous wrapper for async student update call"""
    return asyncio.run(_make_student_update_call_async(givens, conclusion, intermediates, previous_student_response, feedback)) 