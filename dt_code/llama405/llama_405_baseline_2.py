import json
from collections import deque
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from time import sleep

# Add the code directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dt_code.KG.KG_traversal import (
    Neo4jConnection,
    load_nodes_and_parents,
    forward_bfs,
    reconstruct_derivation,
    derive_sequence
)
from dt_code.llm_response_processing.response_preprocess import (
    extract_student_response,
    combine_known_expressions,
    process_student_state,
    extract_next_step,
    extract_json_object,
)
from dt_code.llm_response_processing.step_evaluation import (
    derive_correct_steps,
    check_step,
)
from dt_code.llama405.llama_405_interface import make_student_call, make_teacher_only_call, make_judge_only_call, make_judge_verifier_call, make_student_update_call
# Now you can access the API key as before  

load_dotenv()

URI = os.environ.get('NEO4J_URI')
print(URI)
AUTH = (os.environ.get('NEO4J_USERNAME'), os.environ.get('NEO4J_PASSWORD'))

# Neo4j Connection Class
class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self._driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self):
        self._driver.close()

    def query(self, query, parameters=None):
        with self._driver.session() as session:
            return list(session.run(query, parameters))
        
def append_output_line(outfile, id, problem_number, stepPreState, givens, conclusion, intermediates, correct_step, student_response, judge_response, student_update_response, cleaned_kg_final_steps, completion_tokens, total_tokens, time_taken):
    try:
        # Ensure the output directory exists
        output_dir = Path(outfile).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(outfile, 'a') as f:  # <-- 'a' for append, not 'w'

            output_data = {
                'id': id,
                'problem_number': problem_number,
                'stepPreState': stepPreState,
                'givens': givens,
                'conclusion': conclusion,
                'intermediates': intermediates,
                'correct_step': correct_step,
                'student_response': student_response,
                'judge_response': judge_response,
                'student_update_response': student_update_response,
                'KG_correct_steps': cleaned_kg_final_steps,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'time_taken': time_taken
            }
            f.write(json.dumps(output_data) + '\n')

            f.flush()  # Ensure data is written immediately
            print(f"Successfully appended record {id} to {outfile}")
    except Exception as e:
        print(f"Error writing to file {outfile}: {e}")
        import traceback
        traceback.print_exc()

def main():
    conn = Neo4jConnection(URI, AUTH[0], AUTH[1])
    
    # Cache for derivations to avoid reloading the same cluster
    derivations_cache = {}
    missed_lines = []
    
    
    missed_lines_set = set(missed_lines)
    input_path = 'Data/llm_output/llama405/llama405_baseline_1.jsonl'
    output_path = 'Data/llm_output/llama405/llama405_baseline_2.jsonl'
    
    with open(input_path, 'r') as infile:
        i = 1
        for line in infile:
            line = line.strip()
            
            # Skip empty lines and commented lines
            if not line:
                continue
            if line.startswith('//') or line.startswith('#') or line.startswith('/*'):
                continue
            if line.startswith('*') and line.endswith('*/'):
                continue
                
            try:
                data = json.loads(line)

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:50]}...")
                continue
                
            
            id = data.get('id', '')
            print("id: ", id)
            problem_number = data["problem_number"]
            print("problem number: ", problem_number)
            stepPreState = data.get('stepPreState', '')
            print("step pre state: ", stepPreState)
            givens = data["givens"]
            print("givens: ", givens)
            intermediates = data["intermediates"]
            print("intermediates: ", intermediates)
            intermediates_expressions = intermediates["Expressions"]
            print("intermediates expressions: ", intermediates_expressions)
            known_expressions = givens + intermediates_expressions
            print("known expressions: ", known_expressions)
            correct_step = data["correct_step"]
            print("correct step: ", correct_step)
            conclusion = data["conclusion"]
            print("conclusion: ", conclusion)
            student_response = data["student_response"]
            print("student response: ", student_response)
            
            
            kg_steps, kg_depth = derive_sequence(conn, known_expressions, correct_step, problem_number)   
            print("kg_steps: ", kg_steps)
            print("kg_depth: ", kg_depth)         
            kg_final_steps = derive_correct_steps(kg_steps, known_expressions, correct_step)
            cleaned_kg_final_steps = [s.replace("Step 1: ", "", 1) for s in kg_final_steps]
            print("--------------------------------")
            print("KG Final Steps:")
            print(cleaned_kg_final_steps)
            print("--------------------------------")
            
            
            judge_response_raw, completion_tokens_judge, total_tokens_judge, time_taken_judge = make_judge_only_call(givens, conclusion, intermediates, cleaned_kg_final_steps, student_response)
            judge_response = extract_json_object(judge_response_raw)
            print("--------------------------------")
            print("Judge Response:")
            print(judge_response)
            print("--------------------------------")
            
            if judge_response is None:
                print("Judge response is None")
                print(f"Line {id}")
                missed_lines.append(id)
                i += 1
                continue
            
            # Fix: Judge response is already at top level, not nested under "judge_response"
            # Extract directly from judge_response (not judge_response.get("judge_response"))
            judge_feedback = judge_response.get('JUDGE_FEEDBACK', '')
            student_errors = judge_response.get('STUDENT_ERRORS', '')
            next_step_correctness = judge_response.get('NEXT_STEP_CORRECTNESS', '')
            
            # Format feedback as a clear string message for the student update call
            feedback_message = f"""JUDGE_FEEDBACK: {judge_feedback}
            STUDENT_ERRORS: {student_errors}
            NEXT_STEP_CORRECTNESS: {next_step_correctness}"""
          
            student_update_response_raw, completion_tokens_update, total_tokens_update, time_taken_update = make_student_update_call(givens, conclusion, intermediates, student_response, feedback_message)
            student_update_response = extract_json_object(student_update_response_raw)
            print("--------------------------------")
            print("student update response:")
            print(student_update_response)
            print("--------------------------------")
            if student_update_response is None:
                print("Student update response is None")
                print(f"Line {id}")
                missed_lines.append(id)
                i += 1
                continue
            
            completion_tokens = [completion_tokens_judge, completion_tokens_update]
            total_tokens = [total_tokens_judge, total_tokens_update]
            time_taken = [time_taken_judge, time_taken_update]
            
            #write all the llm responses to the output file here 
            append_output_line(output_path, id, problem_number, stepPreState, givens, conclusion, intermediates, correct_step, student_response, judge_response, student_update_response, cleaned_kg_final_steps, completion_tokens, total_tokens, time_taken)
                                    
            i += 1
            # if i > 50:
            #     break
            print("i: ", i)

    print("Missed lines: ", missed_lines)
if __name__ == "__main__":
    
    main()


