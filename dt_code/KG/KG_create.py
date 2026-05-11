import csv
import json
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError
import os
import sys
from pathlib import Path

# Add the code directory to Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.env_loader import load_env

# Load environment variables
load_env()

# Get Neo4j credentials
URI = os.environ.get('NEO4J_URI')
AUTH = (os.environ.get('NEO4J_USERNAME'), os.environ.get('NEO4J_PASSWORD'))

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    print("Connection established.")

totalNodes = set()
class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self._driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self):
        self._driver.close()

    def query(self, query, parameters=None):
        with self._driver.session() as session:
            return list(session.run(query, parameters))

def clear_database(conn):
    delete_query = "MATCH (n) DETACH DELETE n"
    conn.query(delete_query)
    print("Cleared existing data in the database.")

def parse_derivations(elements):
    derivations = []
    for element in elements:
        node_data = element.split(";")
        # If it's not 'Given', it's a derivation row
        if node_data[2].strip() != "Given":
            derived_expression = node_data[0].strip()
            parents = node_data[1].split('.')
            rule = node_data[2].strip()
            derivations.append((derived_expression, parents, rule))
    return derivations

def create_given_nodes(conn, elements, cluster_id):
    """
    For each row in the CSV, if the third column is 'Given', we MERGE a node labeled :Given
    with an expression and cluster_id.
    """
    for element in elements:
        node_data = element.split(";")
        if node_data[2].strip() == "Given":
            expression = node_data[0].strip()
            query = """
            MERGE (n:Given {expression: $expression, cluster_id: $cluster_id})
            RETURN n
            """
            conn.query(query, parameters={"expression": expression, "cluster_id": cluster_id})

def create_derived_nodes_and_relationships(conn, derivations, cluster_id):
    """
    Creates Derived/Conclusion nodes, each with a node.parents property containing JSON strings.
    Adds cluster_id to each node and relationship.
    """
    for i, (derived_expr, parents, method) in enumerate(derivations):
        node_type = "Conclusion" if i == len(derivations) - 1 else "Derived"
        
        # Prepare the parent set and rule
        # e.g. if parents = ["(A+D)", "-D"] and method="Disjunctive Syllogism",
        # then parent_rule_set = ["(A+D)", "-D", "Disjunctive Syllogism"]
        parents_set = sorted(set(parents)) 
        parent_rule_set = parents_set + [method]

        # Convert the list of parents + rule to a JSON string, e.g. '["(A+D)","-D","Disjunctive Syllogism"]'
        parent_rule_set_json = json.dumps(parent_rule_set)

        # Create or merge the derived node and ensure unique sets of parents/rules appended to n.parents
        derived_query = f"""
        MERGE (n:{node_type} {{expression: $expression, cluster_id: $cluster_id}})
        ON CREATE SET n.parents = []
        ON MATCH SET n.parents = COALESCE(n.parents, [])
        WITH n
        UNWIND n.parents + $new_parent_rule_set AS parent_rule_set
        WITH n, COLLECT(DISTINCT parent_rule_set) AS unique_parents
        SET n.parents = unique_parents
        RETURN n
        """
        
        conn.query(
            derived_query, 
            parameters={
                "expression": derived_expr,
                "cluster_id": cluster_id,
                "new_parent_rule_set": [parent_rule_set_json]
            }
        )

        # Create relationships for each parent
        for parent_expr in parents:
            relationship_query = """
            MATCH (a {expression: $parent_expr, cluster_id: $cluster_id}),
                  (b {expression: $derived_expr, cluster_id: $cluster_id})
            MERGE (a)-[r:DERIVED_BY {method: $method, cluster_id: $cluster_id}]->(b)
            ON CREATE SET r.count = 1
            ON MATCH SET r.count = r.count + 1
            RETURN a, b, r
            """
            conn.query(
                relationship_query,
                parameters={
                    "parent_expr": parent_expr.strip(),
                    "derived_expr": derived_expr,
                    "method": method,
                    "cluster_id": cluster_id
                }
            )

def process_csv_file(file_path, conn):
    """
    Reads a single CSV file and builds/updates the knowledge graph for the specified cluster (derived from file name).
    """
    # Extract cluster_id from the file name (e.g., 'prop_1.1.csv' -> '1.1')
    cluster_id = os.path.splitext(os.path.basename(file_path))[0].split('_')[-1]
    
    try:
        with open(file_path, newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                
                # Track each node expression (just for optional debugging or reference)
                for element in row:
                    newNode = element.split(";")
                    totalNodes.add(newNode[0].strip())
                
                # Create Givens
                create_given_nodes(conn, row, cluster_id)

                # Parse derivations
                derivations = parse_derivations(row)

                # Create derived/conclusion nodes + relationships with parents property
                create_derived_nodes_and_relationships(conn, derivations, cluster_id)

        print(f"Knowledge graph created (or updated) successfully for cluster {cluster_id}.")
    except AuthError as e:
        print(f"Authentication failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_multiple_csv_files(folder_path):
    print(URI, AUTH[0], AUTH[1])
    conn = Neo4jConnection(URI, AUTH[0], AUTH[1])
    clear_database(conn)  
    try:
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                print(f"Processing file: {file_name}")
                process_csv_file(file_path, conn)
    finally:
        conn.close()


def print_all_nodes(conn):
    query = """
    MATCH (n)
    RETURN n.expression as expression, labels(n) as type, n.cluster_id as cluster_id
    """
    results = conn.query(query)
    print("\nAll Nodes in the Graph:")
    print("Expression | Type | Cluster ID")
    print("-" * 50)
    for result in results:
        print(f"{result['expression']} | {result['type']} | {result['cluster_id']}")
        
def count_nodes(conn):
    query = """
    MATCH (n)
    RETURN count(n) as node_count
    """
    results = conn.query(query)
    return results[0]['node_count']



if __name__ == "__main__":
    path = "Data/props"
    # Uncomment this to create the graph
    process_multiple_csv_files(path)
    
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
        print("Connection successful!")
    except Exception as e:
        print(f"Connection failed: {e}")


    conn = Neo4jConnection(URI, AUTH[0], AUTH[1])
    node_count = count_nodes(conn)
    print(f"Total number of nodes in the graph: {node_count}")
    print_all_nodes(conn)