from openai import OpenAI
import os
from dotenv import load_dotenv
from time import sleep
load_dotenv()
import json

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))


def get_correct_step(givens, intermediates, kg_correct_step, next_line_number):
    """
    Calls GPT to generate the next intermediate step in the required format.
    """
    # Format Givens and Intermediates as numbered lists
    # print(givens)
    # print(intermediates)
    # print(kg_correct_step)
    # print(next_line_number)
    
    prompt = f"""
    You are an expert in propositional logic proof formatting.

    Given the following Givens and Intermediates:

    Givens:
    {givens}

    Intermediates:
    {intermediates}

    Your task is to output the next intermediate step, formatted as a single line in the same style, using the information below.

    - The knowledge graph's correct step: "{kg_correct_step}"

    Format your answer as:
    {next_line_number}. (expression) [parent_line_numbers; RuleName]

    For example, if the next step is to derive (P>C) from (-C>-P) using the Contrapositive rule, and (-C>-P) is line 6, your output should be:
    8. (P>C) [6; Contrapositive]

    Respond with only the formatted intermediate step, nothing else.
    """

    # Call GPT (replace with your OpenAI client as needed)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0
    )
    return response.choices[0].message.content


def get_student_step_formatted(givens, intermediates, student_step, student_rule, next_line_number):
    """
    Calls GPT to generate the student's next intermediate step in the required format.
    """
    prompt = f"""
    You are an expert in propositional logic proof formatting.

    Given the following Givens and Intermediates:

    Givens:
    {givens}

    Intermediates:
    {intermediates}

    Your task is to output the student's next intermediate step, formatted as a single line in the same style, using the information below.

    - The student's step: "{student_step}"
    - The student's rule: "{student_rule}"

    Format your answer as:
    {next_line_number}. (expression) [parent_line_numbers; RuleName]

    For example, if the next step is to derive (P>C) from (-C>-P) using the Contrapositive rule, and (-C>-P) is line 6, your output should be:
    8. (P>C) [6; Contrapositive]

    Respond with only the formatted intermediate step, nothing else.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0
    )
    return response.choices[0].message.content

    