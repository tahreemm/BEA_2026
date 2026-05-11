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
    extract_json_object
)
from dt_code.llm_response_processing.step_evaluation import (
    derive_correct_steps,
    check_step,
)
from dt_code.gemini.gemini_interface import make_student_call_gemini, make_teacher_call_gemini, make_judge_call_gemini, make_student_update_call_gemini

# Load environment variables
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
        
def append_output_line(outfile, i, problem_number, givens, conclusion, intermediates, actual_step, student_response, teacher_response, judge_response, student_update_response, student_update_next_step, kg_final_steps, completion_tokens, total_tokens):
    with open(outfile, 'a') as f:  # <-- 'a' for append, not 'w'
        f.write("--------------------------------" + '\n')
        f.write("record: " + str(i) + '\n')
        output_data = {
            'problem_number': problem_number,
            'givens': givens,
            'conclusion': conclusion,
            'intermediates': intermediates,
            'step': actual_step,
            'student_response': student_response,
            'teacher_response': teacher_response,
            'judge_response': judge_response,
            'student_update_response': student_update_response,
            'student_updated_step': student_update_next_step,
            'KG_correct_steps': kg_final_steps,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens
        }
        f.write(json.dumps(output_data) + '\n')
        f.write("--------------------------------" + '\n')
        print("appended line")
        

def main():
    conn = Neo4jConnection(URI, AUTH[0], AUTH[1])
    
    # Cache for derivations to avoid reloading the same cluster
    derivations_cache = {}
    
    input_path = 'Data/cleaned_data/preState.jsonl'
    output_path = 'Data/LLM_output/gemini_output.jsonl'
    
    
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
                
            problem_number, givens, conclusion, intermediates, intermediate_expressions, actual_step, known_expressions = process_student_state(data)
            
            #Use Gemini API calls instead of GPT
            student_response, comp_tokens_student, tot_tokens_student = make_student_call_gemini(givens, conclusion, intermediates)
            student_response = extract_json_object(student_response)
            print("--------------------------------")
            print("Student Response (Gemini):")
            print(student_response)
            print("--------------------------------")
            
            # # Extract the step info from the student response
            # step_info = extract_student_response(student_response)
            # print("--------------------------------")
            # print("student LLM Next Step:")
            # print(step_info)
            # print("--------------------------------")
            # # Load the relevant derivations from the KG
            # derivations = load_nodes_and_parents(conn, problem_number)
            # # Check if the student step is matched with the correct step
            # if_step_correct = check_step(step_info, actual_step, derivations)
            # Derive the correct steps
            student_state = known_expressions
            # target_expr = actual_step
            kg_steps, kg_depth = derive_sequence(conn, known_expressions, actual_step, problem_number)  
            kg_final_steps = derive_correct_steps(kg_steps, student_state, actual_step)
            print("--------------------------------")
            print("KG Final Steps:")
            print(kg_final_steps)
            print("--------------------------------")
            teacher_response, comp_tokens_teacher, tot_tokens_teacher = make_teacher_call_gemini(givens, conclusion, intermediates, actual_step, student_response)
            teacher_response = extract_json_object(teacher_response)
            print("--------------------------------")
            print("Teacher Response (Gemini):")
            print(teacher_response)
            print("--------------------------------")
  
            
            # Call judge LLM here and pass in the teacher response, student response, and correct steps
            judge_response, comp_tokens_judge, tot_tokens_judge = make_judge_call_gemini(givens, conclusion, intermediates, actual_step, student_response, teacher_response, kg_final_steps)
            judge_response = extract_json_object(judge_response)
            judge_feedback = judge_response.get('JUDGE_FEEDBACK', '')
            student_errors = judge_response.get('STUDENT_ERRORS', '')
            # combine judge feedback and student errors as json object
            judge_response_student_errors = {
                'JUDGE_FEEDBACK': judge_feedback,
                'STUDENT_ERRORS': student_errors
            }
            print("--------------------------------")
            print("Judge Response (Llama):")
            print(judge_response)
            print("--------------------------------")
            # sleep(30)
            print("judge response and student errors: ", judge_response_student_errors)
            print("--------------------------------")

            # Call student update LLM here and pass in the student response, judge response, and correct steps
            student_update_response, comp_tokens_update, tot_tokens_update = make_student_update_call_gemini(givens, conclusion, intermediates, student_response, judge_response_student_errors)
            student_update_response = extract_json_object(student_update_response)
            print("--------------------------------")
            print("Student Update Response (Gemini):")
            print(student_update_response)
            print("--------------------------------")

            # Extract the next step from the student update response 
            student_update_next_step = extract_next_step(student_update_response)
            # Collect tokens for this iteration only
            completion_tokens = [comp_tokens_student, comp_tokens_teacher, comp_tokens_judge, comp_tokens_update]
            total_tokens = [tot_tokens_student, tot_tokens_teacher, tot_tokens_judge, tot_tokens_update]
            
            
            # Write all the LLM responses to the output file here 
            
            append_output_line(output_path, i, problem_number, givens, conclusion, intermediates, actual_step, student_response, teacher_response, judge_response, student_update_response, student_update_next_step, kg_final_steps, completion_tokens, total_tokens)
   
            print("i: ", i)
            i += 1
            if i > 486:
                break
    
if __name__ == "__main__":
    main() 