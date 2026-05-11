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
from dt_code.mistral.mistral_interface import make_student_call, make_teacher_only_call, make_judge_only_call, make_judge_verifier_call, make_student_update_call
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
        
def append_output_line(outfile, id, problem_number, stepPreState, givens, conclusion, intermediates, correct_step, student_response, teacher_response, student_update_response, kg_final_steps, completion_tokens, total_tokens, time_taken):
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
                'teacher_response': teacher_response,
                'student_update_response': student_update_response,
                'KG_correct_steps': kg_final_steps,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'time_taken': time_taken
            }
            f.write(json.dumps(output_data) + '\n')
            f.flush()  # Ensure data is written immediately
            print(f"Successfully appended record {problem_number} {id} to {outfile}")
    except Exception as e:
        print(f"Error writing to file {outfile}: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Starting main() function...")
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        print("Attempting to connect to Neo4j...")
        conn = Neo4jConnection(URI, AUTH[0], AUTH[1])
        print("Neo4j connection established successfully")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Cache for derivations to avoid reloading the same cluster
    derivations_cache = {}
    
    input_path = '/Users/tahreemyasir/Documents/prelims/DT_hint-1/Data/cleaned_data/preState.jsonl'
    output_path = 'Data/llm_output/mistral/mistral_baseline_1.jsonl'
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Absolute path would be: {os.path.abspath(input_path)}")
        return
    
    print(f"Input file found: {input_path}")
    
    missed_lines = [198, 391, 406]
    # Set of file line numbers to process (1-indexed)
    missed_lines_set = set(missed_lines)
    
    try:
        with open(input_path, 'r') as infile:
            print("File opened successfully, starting to read lines...")
            i = 1
            # line_num = 0
            for line in infile:
                line = line.strip()
                # line_num += 1  # Track actual file line number
                # if line_num not in missed_lines_set:
                #     continue
                # Skip empty lines and commented lines (but still count them)
                if not line:
                    continue
                if line.startswith('//') or line.startswith('#') or line.startswith('/*'):
                    continue
                if line.startswith('*') and line.endswith('*/'):
                    continue
                # Only process if this file line number is in the target set
                
                print(f"Processing file line (record {i})")
                try:
                    data = json.loads(line)
                    id = data.get('id', '')
                    if id not in missed_lines_set:
                        continue
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON at line: {line[:50]}...")
                    continue
                        
                id, problem_number, givens, conclusion, intermediates, intermediate_expressions, correct_step, known_expressions = process_student_state(data)
                stepPreState = data.get('stepPreState', '')
                
                student_response_raw, completion_tokens_student, total_tokens_student, time_taken_student = make_student_call(givens, conclusion, intermediates)
                student_response = extract_json_object(student_response_raw)
                print("--------------------------------")
                print("Student Response:")
                print(student_response)
                print("--------------------------------")
                sleep(10)
                
                if student_response is None:
                    print("Student response is None")
                    print(f"Line {id}")
                    missed_lines.append(id)
                    i += 1
                    continue
                
                # #derive the correct steps
                student_state = known_expressions
                # target_expr = actual_step
                print("correct_step: ", correct_step)
                print("known_expressions: ", known_expressions)
                kg_steps, kg_depth = derive_sequence(conn, known_expressions, correct_step, problem_number)   
                print("kg_steps: ", kg_steps)
                print("kg_depth: ", kg_depth)         
                kg_final_steps = derive_correct_steps(kg_steps, student_state, correct_step)
                cleaned_kg_final_steps = [s.replace("Step 1: ", "", 1) for s in kg_final_steps]
                print("--------------------------------")
                print("KG Final Steps:")
                print(cleaned_kg_final_steps)
                print("--------------------------------")
                
                
                teacher_response_raw, completion_tokens_teacher, total_tokens_teacher, time_taken_teacher = make_teacher_only_call(givens, conclusion, intermediates, correct_step, student_response)
                teacher_response = extract_json_object(teacher_response_raw)
                print("--------------------------------")
                print("Teacher Response:")
                print(teacher_response)
                print("--------------------------------")
                
                if teacher_response is None:
                    print("Teacher response is None")
                    print(f"Line {id}")
                    missed_lines.append(id)
                    i += 1
                    continue
                #extract the teacher feedback, student errors, and next step correctness from the teacher response
                teacher_feedback = teacher_response.get('TEACHER_FEEDBACK', '')
                student_errors = teacher_response.get('STUDENT_ERRORS', '')
                next_step_correctness = teacher_response.get('NEXT_STEP_CORRECTNESS', '')
                
                # Format feedback as a clear string message for the student update call
                # feedback_message = f"""TEACHER_FEEDBACK: {teacher_feedback}
                #     STUDENT_ERRORS: {student_errors}
                #     NEXT_STEP_CORRECTNESS: {next_step_correctness}"""
                    
                feedback_message = f"""
                    {{
                    "STUDENT_ERRORS": "{student_errors}",
                    "NEXT_STEP_CORRECTNESS": "{next_step_correctness}",
                    "TEACHER_FEEDBACK": "{teacher_feedback}"
                    }}
                    """

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
                
                completion_tokens = [completion_tokens_student, completion_tokens_teacher, completion_tokens_update]
                total_tokens = [total_tokens_student, total_tokens_teacher, total_tokens_update]
                time_taken = [time_taken_student, time_taken_teacher, time_taken_update]
                
                #write all the llm responses to the output file here 
                append_output_line(output_path, id, problem_number, stepPreState, givens, conclusion, intermediates, correct_step, student_response, teacher_response, student_update_response, cleaned_kg_final_steps, completion_tokens, total_tokens, time_taken)
                                        
                i += 1
                # if i >= 50:
                #     break
                print("i: ", i)
                
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"ERROR: An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            conn.close()
            print("Neo4j connection closed")
        except:
            pass
    
    print("Missed lines: ", missed_lines)

if __name__ == "__main__":
    
    main()


