import os
import pandas as pd
import glob
import json
import re

def read_csvs(directory, output_file):
    """
    Reads specific columns from all matching CSV files in the directory and appends them to output_file.
    """
    try:
        # Remove the output file if it exists to avoid appending duplicates
        if os.path.exists(output_file):
            os.remove(output_file)
            
        columns_to_read = [
            "userID",
            "stepPreState",
            "stepPostState",
            "stepHintGivenCount",
            "stepHintType",	
            "stepHintGiven",
            "currentProblem",
            "currentProblemType",
            "currentProblemDescription",
            "currentProblemMetaData",
            "problemHintRequestedCount",
            "problemHintGivenCount",
        ]
        
        first_write = not os.path.exists(output_file)
        
        for i in range(1, 7):
            filename = f"actionLog_L7_{i}_23S.csv"
            filename = os.path.join(directory, filename)
            print("Filename: ", filename)
            if os.path.exists(filename):
                df = pd.read_csv(filename, usecols=columns_to_read, low_memory=False)
                df.to_csv(output_file, mode='a', header=first_write, index=False)
                first_write = False  # Only write header for the first file
            else:
                print(f"File not found: {filename}")
        
    except Exception as e:
        print(f"Error in preprocess_step_student_model_csvs: {str(e)}")
        raise

def clean_and_quote_states(prestate_str):
    # remove trailing /goal
    prestate_str = prestate_str.split("/")[0]
    # split into states
    states = prestate_str.split("],")
    states = [s.strip() + "]" if not s.strip().endswith("]") else s.strip() for s in states]
    
    cleaned = []
    for s in states:
        # remove everything inside brackets [ ... ]
        formula = re.sub(r"\[.*?\]", "", s).strip()
        # wrap in quotes
        cleaned.append(f'"{formula}"')
    
    # join with commas
    return ",".join(cleaned)

def filter_preprocess_csv(input_file, output_file):
    """
    Reads the input_file, retains only rows where sStepCompleted == 1,
    removes duplicates based on sPreState and sPostState,
    and writes the result to output_file.
    """
    try:
        # Remove the output file if it exists to avoid appending duplicates
        if os.path.exists(output_file):
            os.remove(output_file)
        
        columns_to_read = [
            "stepPreState",
            "stepPostState",
            "stepHintGiven",
            "currentProblem",
            "currentProblemType",
            "currentProblemDescription",
            "currentProblemMetaData",
        ]
        df = pd.read_csv(input_file, usecols=columns_to_read, low_memory=False)
        print("raw file shape: ", df.shape)
        filtered_df = df 
        # Drop rows with any NULLs
        
        
        # Drop rows where any cell has '#' in it
        for col in filtered_df.columns:
            mask = filtered_df[col].apply(lambda x: isinstance(x, str) and '#' in x)
            count = mask.sum()
            print(f"Rows in column '{col}' with '#' in it: {count}")
            unique_vals = filtered_df.loc[mask, col].unique()
            if len(unique_vals) > 0:
                print(f"Unique instances in '{col}' with '#' in it: {unique_vals}")
                filtered_df = filtered_df[~mask]
                
        filtered_df = filtered_df.dropna()
        print("raw file shape after deleting rows with any NULLs: ", filtered_df.shape)
        
        # Retain only the rows which have currentProblemType==PS
        filtered_df = filtered_df[filtered_df['currentProblemType'] == 'PS']
        print("filtered file shape after retaining only the rows which have currentProblemType==PS: ", filtered_df.shape)
        
        filtered_df = filtered_df.dropna()
        print("raw file shape after deleting rows with any NULLs: ", filtered_df.shape)
        
        
        # Remove duplicates based on sPreState
        original_shape = filtered_df.shape
        filtered_df = filtered_df.drop_duplicates(subset=['stepPreState'])
        print(f"Removed {original_shape[0] - filtered_df.shape[0]} duplicate rows")
        print("filtered file shape after removing duplicates: ", filtered_df.shape)
        
        # sort the dataframe by currentProblem
        filtered_df = filtered_df.sort_values(by='currentProblem')
        print("filtered file shape after sorting by currentProblem: ", filtered_df.shape)
       
        #remove all the problem where currentProblem has a .8 in it 
        # Define the unwanted suffixes
        unwanted_suffixes = [".8"]
        # Filter out rows where currentProblem ends with any unwanted suffix
        filtered_df = filtered_df[~filtered_df['currentProblem'].astype(str).str.endswith(tuple(unwanted_suffixes))]
        print("filtered file shape after removing all the problem where currentProblem has a .8 in it: ", filtered_df.shape)
        filtered_df.to_csv("Data/cleaned_data/preprocessed_with_postState.csv", index=False)
        
        #remove all the rows where stepHintGiven is "There are no new suggestions at this time."
        filtered_df = filtered_df[filtered_df['stepHintGiven'] != "There are no new suggestions at this time."]
        print("filtered file shape after removing all the rows where stepHintGiven is 'There are no new suggestions at this time.': ", filtered_df.shape)   
        # remove all rows whehre stepHintGiven contains "backward"
        filtered_df = filtered_df[~filtered_df['stepHintGiven'].astype(str).str.contains("backward")]
        print("filtered file shape after removing all the rows where stepHintGiven contains 'backward': ", filtered_df.shape)
        # sort the dataframe by currentProblem
        filtered_df = filtered_df.sort_values(by='currentProblem')
        print("filtered file shape after sorting by currentProblem: ", filtered_df.shape)
        # remove all rows where stepHintGiven contains "Highlight"
        filtered_df = filtered_df[~filtered_df['stepHintGiven'].astype(str).str.contains("Highlight")]
        print("filtered file shape after removing all the rows where stepHintGiven contains 'Highlight': ", filtered_df.shape)
        #remove all rows where stepHintGiven contains "Click on"    
        filtered_df = filtered_df[~filtered_df['stepHintGiven'].astype(str).str.contains("Click on")]
        print("filtered file shape after removing all the rows where stepHintGiven contains 'Click on': ", filtered_df.shape)
        # add another column to the dataframe called "stepHintGiven_cleaned"
        # this columns should be the same as stepHintGiven but with the following changes:
        # remove "Try to derive " from the beginning of the string
        # remove " working forward." from the end of the string
        filtered_df['sAssertion'] = filtered_df['stepHintGiven'].str.replace('^Try to derive ', '', regex=True).str.replace(' working forward.$', '', regex=True)
        print("filtered file shape after adding the stepHintGiven_cleaned column: ", filtered_df.shape)
        # remove all the rows where stepHintGiven = Try to derive the conclusion.
        filtered_df = filtered_df[filtered_df['stepHintGiven'] != "Try to derive the conclusion."]
        print("filtered file shape after removing all the rows where stepHintGiven = 'Try to derive the conclusion.': ", filtered_df.shape)

        filtered_df.to_csv(output_file, index=False)

    except Exception as e:
        print(f"Error in filter_sstepcompleted: {str(e)}")
        raise


def show_problem_type_stats(filtered_file):
    """
    Reads the filtered_file as a dataframe and prints the count of items grouped by currentProblem and currentProblemType.
    """
    df = pd.read_csv(filtered_file)
    print("\nPivot Table (currentProblem as rows, currentProblemType as columns):")
    pivot = df.pivot_table(index='currentProblem', columns='currentProblemType', aggfunc='size', fill_value=0)
    pd.set_option('display.max_rows', None)
    print(pivot)
    
# convert states to ids
mapping_dir = "/Users/tahreemyasir/Documents/prelims/DT_hint-1/Data/map" 
def map_states_to_ids(row):
    problem_num = row["currentProblem"]
    states = [s.strip().strip('"') for s in row["cleanedStates"].split(",")]  # remove quotes for lookup

    mapping_file = os.path.join(mapping_dir, f"mapPropositions_{problem_num}.json")
    if not os.path.exists(mapping_file):
        print(f"⚠️ Mapping file not found for problem {problem_num}")
        return []

    with open(mapping_file) as f:
        mapping = json.load(f)

    ids = []
    for s in states:
        if s in mapping:
            ids.append(mapping[s]["id"])
        else:
            ids.append(None)  # or skip if you prefer
    return ids

# write a function delete the rows where conclusion is part of 



def main():
    try:
        input_directory = "Data/Raw"
        preprocess_file = "Data/cleaned_data/preprocess.csv"
        filtered_file = "Data/cleaned_data/preprocess_filtered.csv"
        
        # Step 1: Preprocess and collect relevant columns
        read_csvs(input_directory, preprocess_file)
        
        # Step 2: Filter rows where sStepCompleted == 1 and remove duplicates
        filter_preprocess_csv(preprocess_file, filtered_file)
        
        print(f"Filtered data saved to {filtered_file}")
        
        df = pd.read_csv(filtered_file)
        # add a column called "cleanedStates" to the dataframe
        df["cleanedStates"] = df["stepPreState"].apply(clean_and_quote_states)
        # add a column called "stateIDs" to the dataframe
        df["stateIDs"] = df.apply(map_states_to_ids, axis=1)
        # Drop rows where any None appears in stateIDs
        df = df[~df["stateIDs"].apply(lambda x: any(i is None for i in x))].reset_index(drop=True)
        # Sort stateIDs in ascending order for each row
        df["stateIDs"] = df["stateIDs"].apply(lambda x: sorted(x))
        # Alternatively, ensure uniqueness by converting to tuples
        df = df[~df["stateIDs"].duplicated(keep="first")].reset_index(drop=True)
        print("df shape after removing duplicates in different order and stateIDs: ", df.shape)
        #save the dataframe to a preprocess_filtered.csv file
        df.to_csv("Data/cleaned_data/preprocess_filtered.csv", index=False)



        
        # # Step 3: Show statistics
        show_problem_type_stats(filtered_file)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()