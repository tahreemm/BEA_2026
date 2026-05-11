import json

def derive_correct_steps(steps, student_state, target_expr):
    print("Steps:")
    print(steps)
    steps_prompt = []
    step_count = 0
    if not steps:
        print(f"No derivation found  with student states {student_state} and target {target_expr}.")
    else:
        print(f"Derivation Steps:")
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
        # print("--------------------------------")
        # print("Step_text:")
        # print(steps_prompt)
        # print("--------------------------------")
        return steps_prompt
    
def check_step(step_info, step, derivations):
    flag = "incorrect"  # Default value
    
    if step_info:
        print("Extracted Step Information:")
        print(f"Step: {step_info['next_step']}")
        print(f"Rule: {step_info['rule']}")
        print(f"Plan: {step_info['plan']}")
        print("--------------------------------")
        
        if step_info['next_step'] == step:
            # print("Correct")
            flag = "correct"
 
        elif step_info['next_step'] in derivations:
            # print("Correct but not optimal")
            flag = "correct but not optimal"

        else:
            # print("Incorrect")
            flag = "incorrect"

    return flag

