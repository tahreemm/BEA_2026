import json
import ast
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from time import sleep
sys.path.append(str(Path(__file__).parent.parent.parent))

from dt_code.GPT.correct_step_formatting import get_correct_step, get_student_step_formatted
from dt_code.llm_response_processing.response_preprocess import extract_student_response, combine_known_expressions
from dt_code.KG.KG_traversal import load_nodes_and_parents, derive_sequence, derive_sequence_with_depth
from dt_code.llm_response_processing.step_evaluation import derive_correct_steps
from dt_code.Llama.Llama_interface import llama_feedback_evaluation
from dt_code.Llama.Llama_interface import llama_helper_eval
# Neo4j Connection Class
load_dotenv()

URI = os.environ.get('NEO4J_URI')
print(URI)
AUTH = (os.environ.get('NEO4J_USERNAME'), os.environ.get('NEO4J_PASSWORD'))

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self._driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self):
        self._driver.close()

    def query(self, query, parameters=None):
        with self._driver.session() as session:
            return list(session.run(query, parameters))

def _ensure_dict(maybe_json_like):
    """
    Normalize an input that may be a dict or a JSON string into a dict.
    Returns an empty dict on failure.
    """
    if isinstance(maybe_json_like, dict):
        return maybe_json_like
    if isinstance(maybe_json_like, str) and maybe_json_like.strip():
        try:
            return json.loads(maybe_json_like)
        except Exception:
            return {}
    return {}

def process_givens(obj):
    """
    Process the givens from the object.
    Returns a list of givens and a list of processed givens lines.
    """
    givens = obj.get('givens', [])
    if isinstance(givens, str):
        try:
            givens_eval = ast.literal_eval(givens)
            if isinstance(givens_eval, list):
                givens = givens_eval
            else:
                givens = [g.strip() for g in givens.split(',') if g.strip()]
        except Exception:
            givens = [g.strip() for g in givens.split(',') if g.strip()]
    processed_givens_lines = []
    for idx, given in enumerate(givens, 1):
        line = f"  {idx}. {given}"
        processed_givens_lines.append(line)
    return givens, processed_givens_lines

def process_intermediates(obj, givens):
    """
    Process the intermediates from the object.
    Returns a list of intermediates and a list of processed intermediates lines.
    """
    intermediates = obj.get('intermediates', {})
    expressions = intermediates.get('Expressions', [])
    rules = intermediates.get('Rules', [])
    start_idx = len(givens) + 1
    processed_intermediates_lines = []
    for idx, (expr, rule) in enumerate(zip(expressions, rules), start=start_idx):
        rule_name = rule
        if isinstance(rule, str) and ';' in rule:
            parts = rule.strip('[]').split(';')
            refs = [p for p in parts[:-1] if not (p.isdigit() and int(p) == idx)]
            rule_name = ';'.join(refs + [parts[-1]]) if refs else parts[-1]
        line = f"  {idx}. {expr}  [{rule_name}]"
        processed_intermediates_lines.append(line)
    return processed_intermediates_lines, start_idx + len(processed_intermediates_lines)

def process_kg_correct_step(obj, processed_givens_lines, processed_intermediates_lines, next_line_number):
    """
    Process the kg correct step from the object.
    Returns a string of the kg correct step.
    """
    kg_steps = obj.get('KG_correct_steps', None)
    correct_step = None
    if kg_steps is not None:
        correct_step = get_correct_step(processed_givens_lines, processed_intermediates_lines, kg_steps, next_line_number)
    if correct_step:
        correct_step_formatted = "  " + correct_step
        return correct_step_formatted
    else:
        return "N/A"

def process_student_response(obj, processed_givens_lines, processed_intermediates_lines, next_line_number):
    try:
        """
        Process the student response from the object.
        Returns a string of the student step and rule.
        """
        student_response = _ensure_dict(obj.get('student_response', {}))
        student_step = student_response.get('NEXT_STEP')
        student_rule = student_response.get('RULE')
        formatted_student_step = get_student_step_formatted(
            processed_givens_lines, processed_intermediates_lines, student_step, student_rule, next_line_number
        )
        if formatted_student_step:
            student_response_formatted = "  " + formatted_student_step
        else:
            student_response_formatted = "  N/A"
        return student_response_formatted
    except Exception:
        student_response_formatted = "  N/A"
        return student_response_formatted

def process_teacher_response(obj):
    try:
        """
        Process the teacher response from the object.
        Returns a string of the teacher rule and feedback.
        """
        teacher_response = _ensure_dict(obj.get('teacher_response', {}))
        return teacher_response.get('TEACHER_RULE'), teacher_response.get('TEACHER_FEEDBACK')
    except Exception:
        return "N/A", "N/A"

def process_judge_response(obj):
    try:
        """
        Process the judge response from the object.
        Returns a string of the judge feedback.
        """
        judge_response = _ensure_dict(obj.get('judge_response', {}))
        return judge_response.get('FINAL_FEEDBACK')
    except Exception:
        return "N/A"

def write_eval_output(filepath, processed_data):
    """
    Write the processed data to the file.
    """
    with open(filepath, 'a') as f:
        # write as jsonl format
        f.write(json.dumps(processed_data) + "\n")
        

def write_processed_output(filepath, processed_data):
    """
    Write the processed data to the file.
    """
    with open(filepath, 'a') as f:
        f.write("\n" + "="*40 + "\n")
        f.write(f"Record number: {processed_data['record_number']}\n")
        f.write(f"Problem number: {processed_data['problem_number']}\n")
        f.write(f"Conclusion: {processed_data['conclusion']}\n\n")
        f.write("Givens:\n")
        for line in processed_data['givens']:
            f.write(f"{line}\n")
        f.write("\nIntermediates and Rules:\n")
        for line in processed_data['intermediates']:
            f.write(f"{line}\n")
        f.write("\nFormatted plan (step and rule) from KG:\n")
        f.write(f"  {processed_data['kg_formatted_step']}\n")
        f.write("\nFormatted step and rule from student:\n")
        f.write(f"  {processed_data['student_formatted_step']}\n")
        f.write("\nActual raw Plan from KG:\n")
        f.write(f"  {processed_data['KG_plan']}\n")
        f.write("\nStudent raw Plan:\n")
        f.write(f"  {processed_data['student_plan']}\n")
        f.write("\nTeacher's Suggestion:\n")
        f.write(f"  {processed_data['teacher_suggestion']}\n")
        f.write("\nJudge's Next Step:\n")
        f.write(f"  {processed_data['judge_suggestion']}\n")
        # write student update plan
        f.write("\nStudent Updated Plan:\n")
        f.write(f"  {processed_data['student_update_plan']}\n")
        f.write("\nFormatted revised step from student:\n")
        f.write(f"  {processed_data['student_revised_step']}\n")
        f.write("\nStudent step distance:\n")
        f.write(f"  {processed_data['student_step_distance']}\n")
        f.write("\nKG step distance:\n")
        f.write(f"  {processed_data['kg_step_distance']}\n")
        
        
import json
import os

def get_use_frequency(student_step, problem_number):
    """
    Get use_frequency for student_step from mapPropositions file.
    Returns use_frequency if found, None otherwise.
    """
    try:
        # Construct file path
        file_path = f"Data/map/mapPropositions_{problem_number}.json"
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
        # Load the JSON file
        with open(file_path, 'r') as f:
            map_data = json.load(f)
        
        # Look for exact match
        if student_step in map_data:
            return map_data[student_step]["use_frequency"]
        
        # If no exact match found
        # print(f"Student step '{student_step}' not found in mapPropositions_{problem_number}.json")
        return None
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

import re

def logical_complexity(expr: str) -> int:
    """
    Scoring rules:
    - Each operator (+, *, =, ->, unary -) counts as 1
    - Each '(' counts as 1
    - Special composites (do NOT also count their operator or '(' separately):
        * '('  -> +2
        - '('  -> +2
        -> '(' -> +4
        = '('  -> +5
    - For '-': count as 1 normally, but if it occurs between 2 operators, count twice
    - Add +1 bonus for boundary between two parenthesized groups: ')<op>('
    """
    if expr is None:
        return 0

    s = str(expr).replace(" ", "")
    n = len(s)
    i = 0
    score = 0

    while i < n:
        # Highest-priority multi-char composite first
        if i + 2 <= n and s[i:i+2] == '>(':
            score += 4  # or 4 if you were using that weight
            i += 2
            continue
        # Two-character composites
        if i + 2 <= n and s[i:i+2] == '*(':
            score += 2
            i += 2
            continue
        # Two-character composites
        if i + 2 <= n and s[i:i+2] == '-(':
            # if '-' is preceded by an operator or ')', treat it as "between operators"
            prev_is_op = (i > 0 and s[i-1] in '+*=-)')
            score += 2 + (1 if prev_is_op else 0)  # base 2, +1 bonus when between ops
            i += 2
            continue
        if i + 2 <= n and s[i:i+2] == '=(':
            score += 5
            i += 2
            continue

        # Bonus for boundary between two parenthesized groups: ')<op>('
        if i + 3 <= n and s[i] == ')' and s[i+2] == '(' and s[i+1] in '*=-':
            score += 1
            # do not consume; let the normal scan count tokens

        # Single tokens and other operators
        ch = s[i]
        if ch == '(':
            score += 1
            i += 1
            continue
        if ch == '+':
            score += 1
            i += 1
            continue
        if ch == '*':
            score += 1
            i += 1
            continue
        if ch == '=':
            score += 1
            i += 1
            continue
        if s[i] == '>':
            score += 1
            i += 1
            continue
        if ch == '-':  # not part of '->(' or '-(' handled earlier
            ops = set('+-*()=')
            prev_is_op = (i > 0 and s[i-1] in ops or s[i-1] == ')')
            next_is_op = (i + 1 < n and (s[i+1] in ops or s[i+1] == '('))
            score += 1 + (2 if (prev_is_op and next_is_op) else 0)
            i += 1
            continue

        # Other characters like variables or ')'
        i += 1

    return score

def extract_evaluation_scores(response_text):
    """
    Extract teacher and judge scores and analyses from an LLM response.
    Returns a dict with parsed scores, analyses, and raw response.
    Uses -1 as fallback for missing or invalid scores.
    """
    def safe_int(value, default=-1):
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    try:
        # Find JSON block within response - more robust regex
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in response")

        json_str = json_match.group()
        scores = json.loads(json_str)

        teacher_score = safe_int(scores.get('TEACHER_SCORE', -1))
        judge_score = safe_int(scores.get('JUDGE_SCORE', -1))
        teacher_correctness_score = safe_int(scores.get('TEACHER_CORRECTNESS_SCORE', -1))
        teacher_correctness_analysis = scores.get('TEACHER_CORRECTNESS_ANALYSIS', "").strip()
        teacher_analysis = scores.get('TEACHER_ANALYSIS', "").strip()
        judge_analysis = scores.get('JUDGE_ANALYSIS', "").strip()
        
        # Extract parent statements for KG plan and student plan
        parent_statements_kg_plan_raw = scores.get('PARENT_STATEMENTS_KG_PLAN', [])
        parent_statements_student_plan_raw = scores.get('PARENT_STATEMENTS_STUDENT_PLAN', [])
        
        # Parse parent statements - handle list, JSON string, or description string
        if isinstance(parent_statements_kg_plan_raw, list):
            parent_statements_kg_plan = parent_statements_kg_plan_raw
        elif isinstance(parent_statements_kg_plan_raw, str) and parent_statements_kg_plan_raw.strip().startswith('['):
            try:
                parent_statements_kg_plan = json.loads(parent_statements_kg_plan_raw)
            except (json.JSONDecodeError, TypeError):
                parent_statements_kg_plan = []
        else:
            parent_statements_kg_plan = []
        
        if isinstance(parent_statements_student_plan_raw, list):
            parent_statements_student_plan = parent_statements_student_plan_raw
        elif isinstance(parent_statements_student_plan_raw, str) and parent_statements_student_plan_raw.strip().startswith('['):
            try:
                parent_statements_student_plan = json.loads(parent_statements_student_plan_raw)
            except (json.JSONDecodeError, TypeError):
                parent_statements_student_plan = []
        else:
            parent_statements_student_plan = []
        
        return {
            'teacher_score': teacher_score,
            'judge_score': judge_score,
            'teacher_correctness_score': teacher_correctness_score,
            'teacher_correctness_analysis': teacher_correctness_analysis,
            'teacher_analysis': teacher_analysis,
            'judge_analysis': judge_analysis,
            'parent_statements_kg_plan': parent_statements_kg_plan,
            'parent_statements_student_plan': parent_statements_student_plan,
        }

    except Exception as e:
        print(f"[extract_evaluation_scores] Error: {e}")
        return {
            'teacher_score': -1,
            'judge_score': -1,
            'teacher_correctness_score': -1,
            'teacher_correctness_analysis': "",
            'teacher_analysis': response_text.strip(),
            'judge_analysis': response_text.strip(),
            'parent_statements_kg_plan': [],
            'parent_statements_student_plan': [],
        }

def main():
    conn = Neo4jConnection(URI, AUTH[0], AUTH[1])
    output_path = 'Data/LLM_cleaned_response/gemini_output_formatted.txt'
    input_path = 'Data/LLM_output/gemini_output.jsonl'
    eval_output_path = 'Data/LLM_eval_data/gemini_output_formatted_eval.jsonl'
    record_number = None
    with open(input_path, 'r') as f:
        i = 1
        for line in f:
            line = line.strip()
            if not line or line.startswith('---'):
                continue
            if line.startswith('record:'):
                # Extract the record number after 'record:'
                record_number = line.split(':', 1)[1].strip()
                continue  # Don't try to parse this line as JSON!
            try:
                obj = json.loads(line)
            except Exception:
                continue
            
            problem_number = obj.get('problem_number')
            print("problem number: ", problem_number)
            conclusion = obj.get('conclusion')
            print("conclusion: ", conclusion)
            correct_step = obj.get('step') # This is the correct step from dataset
            givens = obj.get('givens')
            print("givens: ", givens)
            intermediate_expressions = obj.get('intermediates').get('Expressions', [])
            print("intermediate_expressions: ", intermediate_expressions)
            ###########################
            # derive the derivations from the KG 
            ###########################
            known_expressions = combine_known_expressions(givens, intermediate_expressions)
            print("known_expressions: ", known_expressions)
            length_givens = len(givens)
            print("length of givens: ", length_givens)
            length_intermediate_expressions = len(intermediate_expressions)
            print("length of intermediate_expressions: ", length_intermediate_expressions)
            # print("known_expressions: ", known_expressions)
            derivations = load_nodes_and_parents(conn, problem_number)
            # extract all keys into a list
            derivation_keys = list(derivations.keys())
            # print("derivations keys: ", derivation_keys)
            # remove known_expressions from derivation_keys
            filtered_derivation_keys = [expr for expr in derivation_keys if expr not in known_expressions]
            # print("filtered derivation keys:", filtered_derivation_keys)
            ###########################
            
            # extract the correct step from the KG 
            ############################################################
            KG_plan_raw = obj.get('KG_correct_steps') or obj.get('KG_plan') or ''

            # Normalize container types
            if isinstance(KG_plan_raw, list):
                KG_plan_raw = KG_plan_raw[0] if KG_plan_raw else ''

            s = str(KG_plan_raw).strip()
            # Extract text after first colon if present; otherwise use whole string
            KG_plan = s.split(':', 1)[1].strip() if ':' in s else s
            KG_plan = KG_plan.rstrip("]'\"")
            # Pattern: "Derive (expression) from ... using the RuleName rule."
            step_match = re.search(r'Derive\s+([^from]+)\s+from', KG_plan)
            rule_match = re.search(r'using\s+the\s+([^.]*?)\s+rule', KG_plan)
            KG_step = step_match.group(1).strip() if step_match else "N/A"
            KG_rule = rule_match.group(1).strip() if rule_match else "N/A"
            print("KG step", KG_step)
            KG_complexity = logical_complexity(KG_step)
            print("KG complexity: ", KG_complexity)
            # print("KG rule", KG_rule)
            ###########################################################
            
            # process the givens and intermediates and kg correct step into formatted lines
            givens, processed_givens_lines = process_givens(obj)
            processed_intermediates_lines, next_line_number = process_intermediates(obj, givens)
            # an llm call for formatting the kg correct step
            # kg_formatted_step = process_kg_correct_step(obj, processed_givens_lines, processed_intermediates_lines, next_line_number)
            # process the student initial response and extract the step and rule in the required format
            # formatted_student_step = process_student_response(obj, processed_givens_lines, processed_intermediates_lines, next_line_number)
            #formatting student step to remove spaces and add parentheses if not present
            student_solution = extract_student_response(obj.get('student_response', '')) or {}
            student_step = student_solution.get('next_step', '')
            s = student_step.strip()
            if len(s) >= 3 and not (s.startswith('(') and s.endswith(')')):
                student_step = f"({s})"
            else:
                student_step = s
            student_step = re.sub(r"\s+", "", student_step)
            print("student step: ", student_step)
            student_rule = student_solution.get('rule', '')
            print("student rule: ", student_rule)
            student_plan = student_solution.get('plan', '')
            print("student plan: ", student_plan)
            # depth of the predicted step from the student response
            number_of_steps = derive_sequence_with_depth(conn, known_expressions, student_step, problem_number)
            print("depth for predicted step: ", number_of_steps)
            # derive the distance of the predicted step from the conclusion and the kg step from the conclusion
            kg_known_expressions = known_expressions + [KG_step]
            # print("kg known expressions: ", kg_known_expressions)
            student_known_expressions = known_expressions + [student_step]
            # print("student known expressions: ", student_known_expressions)
            student_step_distance = derive_sequence_with_depth(conn, student_known_expressions, conclusion, problem_number)
            print("distance of the predicted step from the conclusion: ", student_step_distance)
            kg_step_distance = derive_sequence_with_depth(conn, kg_known_expressions, conclusion, problem_number)
            print("distance of the kg step from the conclusion: ", kg_step_distance)
            # calculate the complexity of the predicted step and the kg step
            student_complexity = logical_complexity(student_step)
            print("student complexity: ", student_complexity)
            # get the use frequency of the predicted step
            use_frequency_student = get_use_frequency(student_step, problem_number)
            print("use frequency of the predicted step: ", use_frequency_student)
            use_frequency_kg = get_use_frequency(KG_step, problem_number)
            print("use frequency of the kg step: ", use_frequency_kg)
            # process the teacher response and extract the rule and feedback    
            teacher_rule, teacher_feedback = process_teacher_response(obj)
            judge_suggestion = process_judge_response(obj)

            # Extract from student_updated_step object (student update response) and extract the improved step, better rule, and revised plan
            student_updated_step_raw = obj.get('student_updated_step')

            if isinstance(student_updated_step_raw, str):
                try:
                    student_updated_step_obj = json.loads(student_updated_step_raw)
                except Exception:
                    student_updated_step_obj = {}
            elif isinstance(student_updated_step_raw, dict):
                student_updated_step_obj = student_updated_step_raw
            else:
                student_updated_step_obj = {}
            formatted_student_revised_step = process_student_response(obj, processed_givens_lines, processed_intermediates_lines, next_line_number)
            improved_step = student_updated_step_obj.get('improved_step', None)   # -> null in JSON
            better_rule = student_updated_step_obj.get('better_rule', None)
            revised_plan = student_updated_step_obj.get('revised_plan', None)
            s = improved_step.strip()
            if len(s) >= 3 and not (s.startswith('(') and s.endswith(')')):
                improved_step = f"({s})"
            else:
                improved_step = s
            improved_step = re.sub(r"\s+", "", improved_step)
            print("improved step: ", improved_step)
            use_frequency_improved = get_use_frequency(improved_step, problem_number)
            print("use frequency of the improved step: ", use_frequency_improved)
            # evaluate the teacher response and judge response
            # response_text = llama_feedback_evaluation(KG_plan, student_plan, teacher_feedback, judge_suggestion)
            # result = extract_evaluation_scores(response_text)
            # Access separate components
            # teacher_score = result['teacher_score']
            # print("teacher score: ", teacher_score)
            # judge_score = result['judge_score']
            # print("judge score: ", judge_score)
            # teacher_analysis = result['teacher_analysis']
            # print("teacher analysis: ", teacher_analysis)
            # judge_analysis = result['judge_analysis']
            # print("judge analysis: ", judge_analysis)
            # teacher_correctness_score = result['teacher_correctness_score']
            # print("teacher correctness score: ", teacher_correctness_score)
            # teacher_correctness_analysis = result['teacher_correctness_analysis']     
            # print("teacher correctness analysis: ", teacher_correctness_analysis)
            result = llama_helper_eval(known_expressions, KG_plan, student_plan)
            parent_statements_kg_plan = json.loads(result)['PARENT_STATEMENTS_KG_PLAN']
            print("parent statements kg plan: ", parent_statements_kg_plan)
            parent_statements_student_plan = json.loads(result)['PARENT_STATEMENTS_STUDENT_PLAN']
            print("parent statements student plan: ", parent_statements_student_plan)
            number_of_parent_statements_kg_plan = len(parent_statements_kg_plan)
            print("number of parent statements kg plan: ", number_of_parent_statements_kg_plan)
            number_of_parent_statements_student_plan = len(parent_statements_student_plan)
            print("number of parent statements student plan: ", number_of_parent_statements_student_plan)
            parent_complexities_kg_plan = [logical_complexity(parent) for parent in parent_statements_kg_plan]
            print("parent complexities kg plan: ", parent_complexities_kg_plan)
            parent_complexities_student_plan = [logical_complexity(parent) for parent in parent_statements_student_plan]
            print("parent complexities student plan: ", parent_complexities_student_plan)
            

            # # Prepare data and write to file
            # processed_data = {
            #     'record_number': record_number,
            #     'problem_number': problem_number,
            #     'conclusion': conclusion,
            #     'givens': processed_givens_lines,
            #     'intermediates': processed_intermediates_lines,
            #     'KG_plan': KG_plan,
            #     'kg_formatted_step': kg_formatted_step,
            #     'student_plan': student_plan,
            #     'student_formatted_step': formatted_student_step,
            #     'teacher_suggestion': teacher_feedback,
            #     'judge_suggestion': judge_suggestion,
            #     'student_update_plan': revised_plan,
            #     'student_revised_step': formatted_student_revised_step
            # }
            # write_processed_output(output_path, processed_data)
            
            eval_processed_data = {
                'record_number': record_number,
                'problem_number': problem_number,
                'conclusion': conclusion,
                'givens': processed_givens_lines,
                'intermediates': processed_intermediates_lines,
                'correct_step': correct_step, 
                'known_expressions': known_expressions,
                'length_givens': length_givens,
                'length_intermediate_expressions': length_intermediate_expressions,
                'filtered_derivation_keys': filtered_derivation_keys,
                'KG_plan': KG_plan,
                'kg_rule': KG_rule,
                'kg_step': KG_step,
                'kg_complexity': KG_complexity,
                'student_step': student_step,
                'student_rule': student_rule,
                'student_plan': student_plan,
                'teacher_rule': teacher_rule, #come from teacher response
                'teacher_suggestion': teacher_feedback,
                'judge_suggestion': judge_suggestion, #come from judge response
                'student_updated_rule': better_rule, #come from student update response
                'student_updated_step': improved_step, #come from student update response
                'student_updated_plan': revised_plan,
                'student_step_depth': number_of_steps,
                'student_step_distance': student_step_distance,
                'kg_step_distance': kg_step_distance,
                'student_complexity': student_complexity,
                'use_frequency_student': use_frequency_student,
                'use_frequency_kg': use_frequency_kg,
                'teacher_rule': teacher_rule,
                'teacher_feedback': teacher_feedback,
                'judge_suggestion': judge_suggestion,
                'improved_step': improved_step,
                'better_rule': better_rule,
                'revised_plan': revised_plan, 
                'use_frequency_improved_step': use_frequency_improved,
                'parent_statements_kg_plan': parent_statements_kg_plan,
                'parent_statements_student_plan': parent_statements_student_plan,
                'number_of_parent_statements_kg_plan': number_of_parent_statements_kg_plan,
                'number_of_parent_statements_student_plan': number_of_parent_statements_student_plan,
                'parent_complexities_kg_plan': parent_complexities_kg_plan,
                'parent_complexities_student_plan': parent_complexities_student_plan,
                # 'teacher_score': teacher_score,
                # 'judge_score': judge_score,
                # 'teacher_analysis': teacher_analysis,
                # 'judge_analysis': judge_analysis,
                # 'response_text': response_text,
                # 'teacher_correctness_score': teacher_correctness_score,
                # 'teacher_correctness_analysis': teacher_correctness_analysis,
            }
            
            write_eval_output(eval_output_path, eval_processed_data)
            print("i: ", i)
            i += 1
            sleep(30)
            # if i > 10:
            #     break
if __name__ == "__main__":
    main()
