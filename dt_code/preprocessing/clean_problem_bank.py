import re

def clean_json_file(input_file="Data/Raw/problem_bank.json", output_file="Data/cleaned_data/problem_bank_cleaned.json"):
    try:
        # Read the file
        with open(input_file, 'r') as file:
            content = file.read()
        
        # Pattern to match the fields we want to remove
        pattern = r'"\d+\.\d+":\{"problemCode": "[^"]+","oldProblemCode": "[^"]+","oldCodeTrain": "[^"]+","level": \d+,"difficulty": \d+,'
        
        # Replace with just the problem number and opening brace
        cleaned_content = re.sub(pattern, lambda m: m.group(0).split('{')[0] + '{', content)
        
        # Write to new file
        with open(output_file, 'w') as file:
            file.write(cleaned_content)
        
        print(f"Cleaned content written to {output_file}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    clean_json_file()