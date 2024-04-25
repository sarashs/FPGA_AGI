import os
# decides the recursion limit for the agent graphs
RECURSION_LIMIT = 200

MAX_WEBSEARCH_RESULTS = 1

SERPAPI_PARAMS = {
  "api_key": os.environ['SERPAPI_API_KEY'],
  "engine": "google",
  "q": "to be fillwed",
  "google_domain": "google.com",
  "gl": "us",
  "hl": "en"
}

LANGS = {"sv" : ["systemverilog", "system verilog", "system_verilog", "system-verilog", "sv"],
         "v" : ["v", "verilog"],
         "cpp" : ["hls", "cpp", "c++", "cplusplus", "c plus plus", "c", "hls c", "hls c++", "hls cpp", "vivado hls c++"],
         "vhd": ["vhdl", "vhd", "hdl"], 
         "py": ["python"]}