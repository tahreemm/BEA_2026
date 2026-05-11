import json
import re



def extract_json_object(text):
    """
    Extracts JSON object from text, handling both code blocks and raw JSON.
    Returns the parsed JSON object or None if not found.
    """
    if not text:
        return None
    if isinstance(text, dict):
        return text  # already parsed

    s = text.strip()

    # Try direct parse first
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to find JSON within ```json code blocks
    json_block_start = s.find("```json")
    if json_block_start != -1:
        block_end = s.find("```", json_block_start + 6)
        if block_end != -1:
            fenced_block = s[json_block_start:block_end]
            start = fenced_block.find("{")
            end = fenced_block.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(fenced_block[start:end + 1].strip())
                except Exception:
                    pass

    # Try to find the first {...} block anywhere in the text
    m = re.search(r'\{[\s\S]*\}', s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None
    
def extract_student_response(student_response):
    """
    Extract step information from the student response.
    Returns a dictionary containing UNDERSTANDING, BRAIN_STORMING, PLAN, NEXT_STEP, and RULE.
    Handles empty, malformed, or code-block-wrapped responses robustly.
    """
    if not student_response:
        print("Error: Student response is empty.")
        return None

    # Handle both string and dict inputs
    if isinstance(student_response, dict):
        response_data = student_response
    else:
        if not student_response.strip():
            print("Error: Student response is empty.")
            return None
        try:
            response_data = json.loads(student_response)
        except Exception as e:
            print(f"Error parsing student response: {e}")
            return None

    student_response_info = {
        'understanding': response_data.get('UNDERSTANDING', ''),
        'brain_storming': response_data.get('BRAIN_STORMING', ''),
        'next_step': response_data.get('NEXT_STEP', ''),
        'rule': response_data.get('RULE', ''),
        'plan': response_data.get('PLAN', ''),
        'parent_statements': response_data.get('PARENT_STATEMENTS', ''),
    }
    return student_response_info
    
def combine_known_expressions(givens, intermediate_expressions):
    """
    Combine Givens and intermediate expressions into a single list.
    Returns a list of all known expressions.
    """
    # Convert both to lists if they aren't already
    givens_list = list(givens) if givens else []
    intermediate_list = list(intermediate_expressions) if intermediate_expressions else []
    
    # Combine both lists
    known_expressions = givens_list + intermediate_list
    
    return known_expressions

def process_student_state(data):
    """
    Extract the student state from the data.
    Returns a dictionary containing the problem number, givens, conclusion, intermediates, intermediate expressions, step, and known expressions.
    """
    id = data.get('id', '')
    print("ID:")
    print(id)
    problem_number = data.get('currentProblem')
    print("Problem Number:")
    print(problem_number)
    givens = data.get('Givens')
    print("Givens:")
    print(givens)
    conclusion = data.get('Conclusion')
    print("Conclusion:")
    print(conclusion)
    intermediates = data.get('Intermediates', {})
    intermediate_expressions = intermediates.get('Expressions', [])
    print("Intermediate Expressions:")
    print(intermediate_expressions)
    
    # Combine Givens and intermediate expressions
    known_expressions = combine_known_expressions(givens, intermediate_expressions)
    print("All Known Expressions:")
    print(known_expressions)
    
    step = data.get('sAssertion')
    print("Step:")
    print(step)
    print("--------------------------------")
    return id, problem_number, givens, conclusion, intermediates, intermediate_expressions, step, known_expressions 

def extract_next_step(student_update_response):
    """
    Extract the improved step information from the student update response.
    Returns a dictionary containing the corrected step information.
    """
    if not student_update_response:
        print("Error: Student update response is empty.")
        return None

    # Handle both string and dict inputs
    if isinstance(student_update_response, dict):
        response_data = student_update_response
    else:
        if not student_update_response.strip():
            print("Error: Student update response is empty.")
            return None
        try:
            response_data = json.loads(student_update_response)
        except Exception as e:
            print(f"Error parsing student update response: {e}")
            return None

    next_step_info = {
        'improved_step': response_data.get('IMPROVED_STEP', ''),
        'better_rule': response_data.get('BETTER_RULE', ''),
        'revised_plan': response_data.get('REVISED_PLAN', ''),
    }
    return next_step_info 
    
