# SynthAI: A ReAct-CoT-RAG Framework for Automated Modular HDL/HLS Design Generation

This tool leverages the capabilities of OpenAI's GPT model to automate hardware design based on user prompts. It's an easy-to-use tool that requires minimal inputs from the user.

The tool works by directing the AI to through a waterfall design procedure based on an objective prompt.

## Prerequisites

Ensure you have Python 3.10 installed and the pip package manager.

Before running the project, you'll need to have an OpenAI API key (purchage/get one from here: https://platform.openai.com/). Make sure to set this up, as it is a crucial part of the project.

## Installation

1. Clone the repository:

2. Navigate to the project directory:
    ```
    cd FPGA_AGI
    ```

3. Install the necessary requirements using pip:
    ```
    pip install -r requirements.txt
    ```

## Usage

### Disclaimer: ** The Agents will be able to autonomously generate and run python code on your machine. You are responsible for sandboxing this tool. Run this at your own risk.**

1. Within the root directory add the necessary documents and codes that you want your desginer agent to access to. We currently suppoer pdf and verilog (.v). This is a crucial step to create a knowledge database for RAG. The knowledge database after its creation will be under ./FPGA_AGI/knowledge_base and it will be a chroma DB.

2. Follow the sample jupyter notebook given in the repo.

## License

This software is licensed under the GNU General Public License (GPL), version 3 (GPL-3.0). For the full text of the license, see the [LICENSE](LICENSE) file in this repository.

For more information on GPL-3.0, visit: https://www.gnu.org/licenses/gpl-3.0.en.html

## Status

Please note that this project is currently in its initial stages and under active development. Feedback and contributions are always welcome!
