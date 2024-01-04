from dataclasses import dataclass
import json
import re
from typing import Dict, List, Optional, Any, Union
import os
import warnings

class FormatError(Exception):
    """Exception raised for errors in the input format.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message=None):
        if message is None:
            message = (
                "Input should be of the following form:\n"
                "{\n"
                '    "Module_Name": "name of the module",\n'
                '    "ports": ["specific inputs and outputs, including bit width"],\n'
                '    "description": "detailed description of the module function",\n'
                '    "connections": ["specific other modules it must connect to"],\n'
                '    "hdl_language": "hdl language to be used"\n'
                "}"
            )
        self.message = message
        super().__init__(self.message)


@dataclass
class ProjectDetails:
    goals: str
    requirements: str
    constraints: str

    def save_to_file(self, solution_num):
        dir_path = f'./solution_{solution_num}/'
        with open(os.path.join(dir_path, "GRC.md"), "w+") as file:
            file.write(f"Goals:\n{self.goals}\n\n")
            file.write(f"Requirements:\n{self.requirements}\n\n")
            file.write(f"Constraints:\n{self.constraints}\n")

def fix_json_escapes(json_string):
    # Replace invalid escape sequences
    # Replace \(\ and \)\ with \\(\ and \\)\ respectively
    json_string = re.sub(r'\\(\()', r'\\\\\1', json_string)
    json_string = re.sub(r'\\(\))', r'\\\\\1', json_string)
    # Replace single backslashes with double backslashes, except for valid escape characters
    json_string = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_string)
    return json_string

def extract_json_from_string(string):
    # Extract the JSON part from the string
    json_string = string[string.find('['):string.rfind(']')+1]

    # Convert the JSON string to a Python object
    data = json.loads(json_string)
    
    return data

def extract_project_details(text):
    lines = text.split('\n')
    goals, requirements, constraints = "", "", ""
    current_section = None
    sections = {
        re.compile(r"\**\s*\bGoals\b\s*:*", re.IGNORECASE): "goals",
        re.compile(r"\**\s*\bRequirements\b\s*:*", re.IGNORECASE): "requirements",
        re.compile(r"\**\s*\bConstraints\b\s*:*", re.IGNORECASE): "constraints"
    }
    for line in lines:
        for section_regex in sections:
            if section_regex.match(line):
                current_section = sections[section_regex]
                line = line.split(":", 1)[1] if ":" in line else line  # Remove the header part
                break

        if current_section:
            if current_section == "goals":
                goals += line.strip() + " "
            elif current_section == "requirements":
                requirements += line.strip() + " "
            elif current_section == "constraints":
                constraints += line.strip() + " "

    return ProjectDetails(goals.strip(), requirements.strip(), constraints.strip())

def lang2suffix(lang):
  if lang.lower() in ["systemverilog", "system verilog", "system_verilog", "system-verilog", "sv"]:
    return ".sv"
  if lang.lower() in ["v", "verilog"]:
    return ".v"
  if lang.lower() in ["hls", "cpp", "c++", "cplusplus", "c plus plus", "c", "hls c", "hls c++", "hls cpp", "vivado hls c++"]:
    return ".cpp"
  if lang.lower() in ["vhdl", "vhd", "hdl"]:
    return ".hdl"

LANGS = (["systemverilog", "system verilog", "system_verilog", "system-verilog", "sv"] +
         ["v", "verilog"] +
         ["hls", "cpp", "c++", "cplusplus", "c plus plus", "c", "hls c", "hls c++", "hls cpp", "vivado hls c++"] +
         ["vhdl", "vhd", "hdl"] + ["python"])

def extract_codes_from_string(string):
    lower_string = string.lower()  # Convert the entire string to lowercase
    code = None
    for lang in LANGS:
        # Create the search pattern for each language in lowercase
        start_pattern = f'```{lang.lower()}\n'
        end_pattern = '\n```'

        # Find the start and end indices for each code block
        start_index = lower_string.find(start_pattern)
        end_index = lower_string.rfind(end_pattern)

        if start_index != -1 and end_index != -1:
            # Calculate the actual start index in the original string
            actual_start_index = start_index + len(start_pattern)

            # Extract and store the code block from the original string
            code = string[actual_start_index:end_index]
            break

    # Check if any code blocks have been extracted
    if not code:
        # If no code blocks are found, return the original string
        return string
    else:
        return code

def extract_json_from_string(string):
    # Extract the JSON part from the string
    if string.find('```json\n') != -1:
        json_string = string[string.find('```json\n')+8:string.rfind('\n```')]
    else:
        json_string = string
    # Convert the JSON string to a Python object
    data = json.loads(fix_json_escapes(json_string))
    return data

def save_solution(codes: List, solution_num: int = 0):
    """ Saves the generated solution """
    for module, code in codes:
        suffix = lang2suffix(module["hdl_language"])
        dir_path = f'./solution_{solution_num}/'
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            #shutil.rmtree(dir_path)
            pass
        else:
            os.makedirs(dir_path)
        with open(os.path.join(dir_path, module["Module_Name"] + suffix), "w+") as file:
            content = code
            file.write(content)
    print("The solution was saved successfully!")