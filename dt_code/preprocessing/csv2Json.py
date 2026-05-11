import csv
import json
import re
import sys
from pathlib import Path
import pandas as pd

# Add the project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dt_code.llm_response_processing.response_preprocess import process_student_state
    
def parse_sPreState(sPreState):
    if '/' in sPreState:
        rule_part, conclusion = sPreState.rsplit('/', 1)
        #print("conclusion: ", conclusion)
    else:
        rule_part = sPreState
        conclusion = ""

    # Split by commas but keep parentheses and brackets together
    entries = [e.strip() for e in re.split(r'(?<=\])', rule_part) if e.strip()]
    #print("entries: ", entries)
    givens = []
    expressions = []
    rules = []

    for entry in entries:
        entry = entry.lstrip(',')
        #print("entry: ", entry)
        match = re.split(r'(?=\[)', entry)
        #print("match: ", match)
        if "Given" in match[1]:
            givens.append(match[0])
        else: 
            #print("match: ", match)
            expressions.append(match[0])
            rules.append(match[1])

        
    return {
         "Givens": givens,
         "Intermediates": {
             "Expressions": expressions,
             "Rules": rules
         },
         "Conclusion": conclusion.strip()
    }

def convert_csv_to_jsonl(csv_file_path, jsonl_file_path):
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile, open(jsonl_file_path, 'w', encoding='utf-8') as jsonlfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            entry = {
                "currentProblem": row["currentProblem"],
                "stepPreState": row["stepPreState"],
                "currentProblemDescription": row["currentProblemDescription"],
                "currentProblemMetaData": row["currentProblemMetaData"],
                "sAssertion": row["sAssertion"]
            }
            spre = parse_sPreState(row["stepPreState"])
            entry.update(spre)
            jsonlfile.write(json.dumps(entry) + '\n')

def add_KG_hint(jsonl_file_path):
    with open(jsonl_file_path, 'r') as jsonlfile:
        for line in jsonlfile:
            entry = json.loads(line)
            problem_number, givens, conclusion, intermediates, intermediate_expressions, step, known_expressions = process_student_state(entry)
            entry["KG_hint"] = "KG_hint"
            jsonlfile.write(json.dumps(entry) + '\n')

def remove_rows_where_conclusion_in_expressions(jsonl_file_path):
    """
    Remove rows where conclusion appears as a separate expression in the expressions list
    (not as a substring within another expression)
    """
    # Read all lines
    with open(jsonl_file_path, 'r') as f:
        lines = f.readlines()
    
    filtered_lines = []
    removed_count = 0
    
    for line in lines:
        try:
            entry = json.loads(line.strip())
            
            # Get conclusion and expressions
            conclusion = entry.get('Conclusion', '').strip()
            expressions = entry.get('Intermediates', {}).get('Expressions', [])
            
            # Check if conclusion appears as a separate expression (exact match)
            conclusion_as_separate_expression = conclusion in expressions
            
            if not conclusion_as_separate_expression:
                filtered_lines.append(line)
            else:
                print(f"Removed row where conclusion is a separate expression: {line}")
                removed_count += 1
                
        except json.JSONDecodeError:
            # Keep lines that can't be parsed
            filtered_lines.append(line)
    
    # Write filtered lines back to file
    with open(jsonl_file_path, 'w') as f:
        f.writelines(filtered_lines)
    
    print(f"Removed {removed_count} rows where conclusion is a separate expression")
    print(f"Remaining rows: {len(filtered_lines)}")


def show_pivot_table_by_problem(jsonl_file_path):
    """
    Show pivot table based on currentProblem
    """
    # Read the JSONL file and extract data
    data = []
    
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                data.append({
                    'currentProblem': entry.get('currentProblem', ''),
                    'sAssertion': entry.get('sAssertion', ''),
                    'problemDescription': entry.get('currentProblemDescription', '')
                })
            except json.JSONDecodeError:
                continue
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create pivot table
    pivot_table = df.groupby('currentProblem').agg({
        'sAssertion': 'count'
    }).reset_index()
    
    pivot_table.columns = ['Problem_Number', 'Instance_Count']
    
    # Sort by problem number
    pivot_table = pivot_table.sort_values('Problem_Number')
    
    print("Pivot Table - Problem Numbers and Instance Counts:")
    print(pivot_table.to_string(index=False))
    
    # Show summary
    print(f"\nTotal problems: {len(pivot_table)}")
    print(f"Total instances: {pivot_table['Instance_Count'].sum()}")
    
    return pivot_table


def main():
    csv_file = "Data/cleaned_data/preprocess_filtered.csv"     # Update if needed
    jsonl_file = "Data/cleaned_data/preState.jsonl"
    convert_csv_to_jsonl(csv_file, jsonl_file)
    remove_rows_where_conclusion_in_expressions("Data/cleaned_data/preState.jsonl")
    
    # Add this line to print the pivot table
    show_pivot_table_by_problem("Data/cleaned_data/preState.jsonl")

    

if __name__ == "__main__":
    main()
