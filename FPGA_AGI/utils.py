from dataclasses import dataclass
import json
import re
from typing import Dict, List, Optional, Any, Union

@dataclass
class ProjectDetails:
    goals: str
    requirements: str
    constraints: str

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
        re.compile(r"[-]*\s*Goals[:]*"): "goals",
        re.compile(r"[-]*\s*Requirements[:]*"): "requirements",
        re.compile(r"[-]*\s*Constraints[:]*"): "constraints"
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
  
def extract_codes_from_string(string):
    # Extract the JSON part from the string
    json_string = string[string.find('```json\n')+8:string.rfind('\n```')]
    # Convert the JSON string to a Python object
    data = json.loads(json_string)
    return data

def save_solution(codes: List, solution_num: int = 0):
    """ Saves the generated solution """
    for item in codes:
        code = extract_codes_from_string(item)
        module = code[list(code.keys())[0]]
        suffix = lang2suffix(module["hdl_language"])
        with open(f'./solution_{solution_num}/'+ list(code.keys())[0] + suffix, "w") as file:
            content = module["code"]
            file.write(content)
    print("The solution was saved successfully!")