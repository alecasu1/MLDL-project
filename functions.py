
import re

def change_var_sh(file_path, var_name, new_value):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            if line.startswith(f'export {var_name}='):
                new_line = re.sub(f'export {var_name}=.*', f'export {var_name}={new_value}', line)
                file.write(new_line + '\n')
            else:
                file.write(line)

def change_layers(class_content):
    # Define the function logic here
    pass

# Add other function definitions here
