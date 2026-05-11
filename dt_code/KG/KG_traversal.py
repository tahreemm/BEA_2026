import json
from collections import deque
from neo4j import GraphDatabase
from openai import OpenAI
import os
import sys
from pathlib import Path

# Add the code directory to Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.env_loader import load_env

# Load environment variables
load_env()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Get Neo4j credentials
URI = os.environ.get('NEO4J_URI')
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

def load_nodes_and_parents(conn, cluster_id):
    """
    Returns a dict of { expression: [ [p1, p2, ..., method], ... ] } for a specific cluster.
    Each sub-list is a possible derivation route (parents + rule).
    """
    conn = Neo4jConnection(URI, AUTH[0], AUTH[1])
    try:
        query = """
        MATCH (n {cluster_id: $cluster_id})
        RETURN n.expression AS expr, n.parents AS parentList
        """
        results = conn.query(query, {"cluster_id": cluster_id})

        node_derivations = {}
        for record in results:
            expr = record["expr"]
            parentList = record["parentList"]  
            if parentList is None:
                node_derivations[expr] = []  # No parents for 'Given' nodes
            else:
                all_derivations = []
                for json_str in parentList:
                    deriv = json.loads(json_str)  
                    all_derivations.append(deriv)
                node_derivations[expr] = all_derivations
                
        return node_derivations
    finally:
        conn.close()

# BFS LOGIC (Cluster-Specific)
def forward_bfs(conn, student_states, target_expression, cluster_id):
    """
    Perform BFS from known states to derive the target within the specified cluster.
    Only nodes belonging to the given cluster_id are considered.
    """
    node_derivations = load_nodes_and_parents(conn, cluster_id)  # Load only cluster-specific nodes

    discovered = set(student_states)
    queue = deque(student_states)

    derived_map = {}
    depth_map = {}

    for expr in student_states:
        derived_map[expr] = {"used_parents": [], "method": "Given"}  # Known states are 'Given'
        depth_map[expr] = 0

    while queue:
        current = queue.popleft()
        if current == target_expression:
            # Return depth for the target when found
            return derived_map, discovered, True, depth_map.get(target_expression, -1)

        # Check if current node helps derive other nodes
        for child_expr, derivation_list in node_derivations.items():
            if child_expr in discovered:
                continue  # Already derived

            for parent_set in derivation_list:
                if not parent_set:
                    continue

                potential_parents = parent_set[:-1]  # All parents
                method = parent_set[-1]  # Derivation rule

                # Check if all parents in this set are already discovered
                if all(p in discovered for p in potential_parents):
                    discovered.add(child_expr)
                    derived_map[child_expr] = {
                        "used_parents": list(potential_parents),
                        "method": method
                    }
                    depth_map[child_expr] = depth_map[current] + 1
                    queue.append(child_expr)
                    break  

    
    return derived_map, discovered, (target_expression in discovered), depth_map.get(target_expression, -1)

# RECONSTRUCTING THE PATH
def reconstruct_derivation(derived_map, student_states, target_expression):
    """
    Once BFS is done, reconstruct the derivation steps ensuring parents come before children.
    """
    if target_expression not in derived_map:
        return []  

    visited = set()
    order = []

    def dfs(expr):
        if expr in visited:
            return
        visited.add(expr)
        parents = derived_map[expr]["used_parents"]
        for p in parents:
            dfs(p)
        order.append(expr)

    dfs(target_expression)

    # Create final step list
    step_list = []
    for expr in order:
        info = derived_map[expr]
        used_parents = info["used_parents"]
        method = info["method"]
        if method == "Given":
            step_list.append([expr, [None]])
        else:
            if len(used_parents) == 1:
                step_list.append([expr, [used_parents[0], method]])
            else:
                step_list.append([expr, used_parents + [method]])
    # print("step_list: ", step_list)
    return step_list

# Putting it all together
def derive_sequence(conn, student_states, target_expression, cluster_id):
    """
    Derive the sequence of steps from known states to the target expression within the cluster.
    """
    derived_map, discovered, success, depth = forward_bfs(conn, student_states, target_expression, cluster_id)
    if not success:
        return [], depth

    steps = reconstruct_derivation(derived_map, student_states, target_expression)

    return steps, depth

def derive_sequence_with_depth(conn, student_states, target_expression, cluster_id):
    """
    Derive the sequence of steps from known states to the target expression within the cluster.
    """
    try:
        cluster_id = cluster_id
        student_state = student_states
        target_expr = target_expression
    
        steps, depth = derive_sequence(conn, student_state, target_expr, cluster_id)

        steps_prompt = []
        step_count = 0
        if not steps:
            # print(f"No derivation found: ", depth)
            print("")
        else:
            # print(f"Derivation Steps for question")
            for step in steps:
                # print(step)
                derived_expression = step[0]
                derivation_info = step[1]

                if derivation_info[-1] is None:
                    continue
                
                step_count += 1
                
                # Extract derivation details
                method = derivation_info[-1]  
                parents = derivation_info[:-1]  

                if len(parents) == 1:
                    step_text = f"Step {step_count}: Derive {derived_expression} from {parents[0]} using the {method} rule."
                elif len(parents) == 2:
                    step_text = (f"Step {step_count}: Derive {derived_expression} from {parents[0]} and {parents[1]} "
                                    f"using the {method} rule.")
                else:
                    step_text = f"Step {step_count}: Derive {derived_expression} using the {method} rule."

                steps_prompt.append(step_text)

            final_prompt = "\n".join(steps_prompt)
            # print("\nGenerated Prompt for LLM:")
            # print(final_prompt)
            # print("steps prompt: ", len(steps_prompt))
            return len(steps_prompt)
    finally:
        conn.close()

        
    
    
def main():
    #    number_of_questions = 1
#    cluster_ids = ["6.6", "6.6"]
#    student_states = [["(Y=P)", "(-Y>-C)", "(-P=-C)", "((-P>-C)*(-C>-P))", "((Y>P)*(P>Y))"], 
#                      ["(Y=P)", "(-Y>-C)", "(-P=-C)", "((-P>-C)*(-C>-P))", "((Y>P)*(P>Y))"]]
#    target_expressions = ["(Y>P)", "(Y>P)"]
#    for question in range(number_of_questions):
#        cluster_id = cluster_ids[question]
#        student_state = student_states[question]
#        target_expr = target_expressions[question]
#        number_of_steps = derive_sequence_with_depth(conn, student_state, target_expr, cluster_id)
#        print("number_of_steps: ", number_of_steps)
    conn = Neo4jConnection(URI, AUTH[0], AUTH[1])
    try:
        # cluster_id = input("Enter the cluster ID (e.g., '1.1'): ").strip()
        # student_state = input("Enter the known states (comma-separated): ").strip().split(",")
        # target_expr = input("Enter the target expression: ").strip()
        number_of_questions = 3
        cluster_ids = ["5.6", "6.6", "6.6"]
        student_states = [["(K>M)", "(Z>R)", "-(K>R)", "(-R>-Z)", "-(-R>-K)"], 
                          ["(Y=P)", "(-Y>-C)", "(-P=-C)", "((-P>-C)*(-C>-P))", "((Y>P)*(P>Y))"], 
                     ["(Y=P)", "(-Y>-C)", "(-P=-C)", "((-P>-C)*(-C>-P))", "((Y>P)*(P>Y))"]]
        target_expressions = ["(K*-R)", "(Y>P)", "(Y>P)"]

        for question in range(number_of_questions):
            
            cluster_id = cluster_ids[question]
            student_state = student_states[question]
            target_expr = target_expressions[question]
        
            steps, depth = derive_sequence(conn, student_state, target_expr, cluster_id)
            print("steps: ", len(steps))
            steps_prompt = []
            step_count = 0
            if not steps:
                print(f"No derivation found for question {question+1} with student states {student_state} and target {target_expr}.")
            else:
                print(f"Derivation Steps for question {question+1}:")
                for step in steps:
                    print(step)
                    derived_expression = step[0]
                    derivation_info = step[1]

                    if derivation_info[-1] is None:
                        continue
                    
                    step_count += 1
                    
                    # Extract derivation details
                    method = derivation_info[-1]  
                    parents = derivation_info[:-1]  

                    if len(parents) == 1:
                        step_text = f"Step {step_count}: Derive {derived_expression} from {parents[0]} using the {method} rule."
                    elif len(parents) == 2:
                        step_text = (f"Step {step_count}: Derive {derived_expression} from {parents[0]} and {parents[1]} "
                                     f"using the {method} rule.")
                    else:
                        step_text = f"Step {step_count}: Derive {derived_expression} using the {method} rule."

                    steps_prompt.append(step_text)

                final_prompt = "\n".join(steps_prompt)
                print("\nGenerated Prompt for LLM:")
                print(final_prompt)
                print("steps prompt: ", len(steps_prompt))
                print("depth: ", depth)

    finally:
        conn.close()
       
if __name__ == "__main__":
    main()


