import json
import ast
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from time import sleep
import pandas as pd
sys.path.append(str(Path(__file__).parent.parent.parent))

from dt_code.GPT.correct_step_formatting import get_correct_step, get_student_step_formatted
from dt_code.llm_response_processing.response_preprocess import extract_student_response, combine_known_expressions, extract_json_object
from dt_code.KG.KG_traversal import load_nodes_and_parents, derive_sequence, derive_sequence_with_depth
from dt_code.llm_response_processing.step_evaluation import derive_correct_steps
from dt_code.GPT.csv_baseline_1 import get_student_step_depth, logical_complexity, convert_rule_to_short_name
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
    
def process_student_updated_response(obj):
    """
    Process the student updated response from the object.
    Returns a tuple containing (improved_step, better_rule, revised_plan).
    Returns (None, None, None) if extraction fails.
    """
    try:
        # The key in the JSON is 'student_update_response' (without 'd')
        student_update_response = _ensure_dict(obj.get('student_update_response', {}))
        improved_step = student_update_response.get('IMPROVED_STEP')
        better_rule = student_update_response.get('BETTER_RULE')
        revised_plan = student_update_response.get('REVISED_PLAN')
        return improved_step, better_rule, revised_plan
    except Exception as e:
        print(f"Error in process_student_updated_response: {e}")
        return None, None, None


def process_teacher_hint_response(obj):
    try:
        """
        Process the teacher response from the object.
        Returns a string of the teacher rule and feedback.
        """
        teacher_response = _ensure_dict(obj.get('teacher_response_hint', {}))
        return teacher_response.get('TEACHER_PLAN'), teacher_response.get('TEACHER_RULE'), teacher_response.get('TEACHER_PARENT_STATEMENTS'), teacher_response.get('STUDENT_ERRORS'), teacher_response.get('NEXT_STEP_CORRECTNESS'), teacher_response.get('TEACHER_FEEDBACK')
    except Exception:
        return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
    
def process_teacher_base_response(obj):
    try:
        """
        Process the teacher response from the object.
        Returns a string of the teacher rule and feedback.
        """
        teacher_response = _ensure_dict(obj.get('teacher_response_base', {}))
        return teacher_response.get('TEACHER_NEXT_STEP'), teacher_response.get('TEACHER_PLAN'), teacher_response.get('TEACHER_RULE'), teacher_response.get('TEACHER_PARENT_STATEMENTS'), teacher_response.get('STUDENT_ERRORS'), teacher_response.get('TEACHER_FEEDBACK')
    except Exception:
        return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"

def process_judge_response(obj):
    try:
        """
        Process the judge response from the object.
        Returns a string of the teacher plan errors, student errors, teacher assessment errors, teacher feedback correctness, and judge feedback.
        """
        judge_response = _ensure_dict(obj.get('judge_response', {}))
        return judge_response.get('TEACHER_PLAN_ERRORS'), judge_response.get('STUDENT_ERRORS'), judge_response.get('TEACHER_ASSESSMENT_ERRORS'), judge_response.get('TEACHER_FEEDBACK_CORRECTNESS'), judge_response.get('JUDGE_FEEDBACK')
    except Exception:
        return "N/A", "N/A", "N/A", "N/A", "N/A"

def write_eval_output(filepath, processed_data):
    """
    Write the processed data to the file.
    """
    with open(filepath, 'a') as f:
        # write as jsonl format
        f.write(json.dumps(processed_data) + "\n")
        
def write_accuracy_output(filepath, processed_data):
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

def main():
    conn = Neo4jConnection(URI, AUTH[0], AUTH[1])
    input_path = 'Data/llm_output/gpt_baseline_2.jsonl'
    record_number = None
    all_rows = []
    with open(input_path, 'r') as f:
        i = 1
        record_number = i
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
            row_data = {}
            problem_number = obj.get('problem_number')
            print("problem number: ", problem_number)
            conclusion = obj.get('conclusion')
            print("conclusion: ", conclusion)
            correct_step = obj.get('step') # This is the correct step from dataset
            givens = obj.get('givens')
            print("givens: ", givens)
            stepPreState = obj.get('stepPreState')
            print("stepPreState: ", stepPreState)
            intermediate_expressions = obj.get('intermediates').get('Expressions', [])
            print("intermediate_expressions: ", intermediate_expressions)

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
            row_data["record_number"] = record_number
            row_data["problem_number"] = problem_number
            row_data["conclusion"] = conclusion
            row_data["givens"] = givens
            row_data["intermediate_expressions"] = intermediate_expressions
            row_data["known_expressions"] = known_expressions
            row_data["length_givens"] = length_givens
            row_data["length_intermediate_expressions"] = length_intermediate_expressions
            row_data["filtered_derivation_keys"] = filtered_derivation_keys
            
            # KG information
            KG_text = (obj.get("KG_correct_steps") or [""])[0]
            KG_plan = re.sub(r'^Step\s*\d+\s*:\s*', '', KG_text)
            print("KG plan: ", KG_plan)
            # Pattern: "Derive (expression) from ... using the RuleName rule."
            plan_match = re.search(r'Derive\s+([^from]+)\s+from', KG_plan)
            rule_match = re.search(r'using\s+the\s+([^.]*?)\s+rule', KG_plan)
            KG_step = plan_match.group(1).strip() if plan_match else "N/A"
            KG_rule = rule_match.group(1).strip() if rule_match else "N/A"
            # Convert full rule name to short name
            KG_rule = convert_rule_to_short_name(KG_rule)
            print("KG rule: ", KG_rule)
            print("KG step", KG_step)
            KG_complexity = logical_complexity(KG_step)
            print("KG complexity: ", KG_complexity)
            # derive the distance of the predicted step from the conclusion and the kg step from the conclusion
            kg_known_expressions = known_expressions + [KG_step]
            kg_step_distance = derive_sequence_with_depth(conn, kg_known_expressions, conclusion, problem_number)
            print("distance of the kg step from the conclusion: ", kg_step_distance)
            use_frequency_kg = get_use_frequency(KG_step, problem_number)
            print("use frequency of the kg step: ", use_frequency_kg)
            KG_depth = derive_sequence_with_depth(conn, known_expressions, KG_step, problem_number)
            print("KG depth: ", KG_depth)

            row_data["KG_plan"] = KG_plan
            row_data["KG_rule"] = KG_rule
            row_data["KG_complexity"] = KG_complexity
            row_data["KG_step"] = KG_step
            row_data["KG_distance"] = kg_step_distance
            row_data["KG_frequency"] = use_frequency_kg
            row_data["KG_depth"] = KG_depth
            
            student_response = obj.get("student_response", "") or {}
            student_plan = student_response.get("REASONING")
            print("student plan: ", student_plan)
            student_step = student_response.get("NEXT_STEP")
            print("student step: ", student_step)
            student_rule = student_response.get("RULE")
            print("student rule: ", student_rule)
            student_step_depth, student_step_distance = get_student_step_depth(conn, problem_number, student_step, givens, intermediate_expressions, known_expressions, conclusion)
            print("student depth: ", student_step_depth)
            print("student step distance: ", student_step_distance)

            # 5. Compute additional metrics
            student_complexity = logical_complexity(student_step)
            print("student complexity: ", student_complexity)
            use_frequency_student = get_use_frequency(student_step, problem_number)
            print("use frequency student: ", use_frequency_student)
            student_parent_statements = student_response.get("PARENT_STATEMENTS", [])
            print("student parent statements: ", student_parent_statements)
            student_parent_complexities = [logical_complexity(parent) for parent in student_parent_statements]
            print("student parent complexities: ", student_parent_complexities)
            # Get completion and total tokens using the student response index
            
            completion_tokens = obj.get("completion_tokens", "")    
            completion_time = obj.get("time_taken", "")
            row_data["student_plan"] = student_plan
            row_data["student_step"] = student_step
            row_data["student_rule"] = student_rule
            row_data["student_complexity"] = student_complexity
            row_data["student_depth"] = student_step_depth
            row_data["student_distance"] = student_step_distance
            row_data["student_frequency"] = use_frequency_student
            row_data["student_parent_statements"] = student_parent_statements
            row_data["student_parent_complexities"] = student_parent_complexities
            row_data["student_completion_tokens"] = [completion_tokens[0]] 
            row_data["student_completion_time"] = [completion_time[0]] 
            # judge response
            judge_response = obj.get("judge_response", "")
            next_step_correctness = judge_response.get("NEXT_STEP_CORRECTNESS")
            print("next step correctness: ", next_step_correctness)
            judge_feedback = judge_response.get("JUDGE_FEEDBACK")
            print("judge feedback: ", judge_feedback)
            row_data["next_step_correctness"] = next_step_correctness
            row_data["judge_feedback"] = judge_feedback
            row_data["judge_completion_tokens"] = [completion_tokens[1]] 
            row_data["judge_completion_time"] = [completion_time[1]]   


            student_update_response = obj.get("student_update_response") or {}
            student_update_plan = student_update_response.get("IMPROVED_STEP", None)
            print("student update plan: ", student_update_plan)
            student_update_rule = student_update_response.get("BETTER_RULE", None)
            print("student update rule: ", student_update_rule)
            student_update_revised_plan = student_update_response.get("REVSED_PLAN", None)
            print("student update revised plan: ", student_update_revised_plan)
            row_data["student_update_plan"] = student_update_plan
            row_data["student_update_rule"] = student_update_rule
            row_data["student_update_revised_plan"] = student_update_revised_plan
            # row_data["student_update_completion_tokens"] = [completion_tokens[2]] 
            # print("student update completion tokens: ", completion_tokens[2])
            # row_data["student_update_completion_time"] = [completion_time[2]]   
            # print("student update completion time: ", completion_time[2])
            
            
            all_rows.append(row_data)
        
            print("i: ", i)
            i += 1
            # if i > 6:
            #     break
        df = pd.DataFrame(all_rows)
        print(df.shape)
        df.to_csv("Data/csv/gpt_baseline_2.csv", index=False)
        print("CSV saved successfully!")
            
if __name__ == "__main__":
    main()
