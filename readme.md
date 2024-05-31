# SynthAI: A Multi Agent Generative AI Framework for Automated Modular HLS Design Generation

This tool leverages the capabilities of OpenAI's GPT model to automate hardware design based on user prompts. It's an easy-to-use tool that requires minimal inputs from the user.

The tool works by directing the AI to through a waterfall design procedure based on a set of goals and requirements. Follow this link for an updated version of the accompanying paper: https://arxiv.org/abs/2405.16072

## Prerequisites

Ensure you have Python 3.10 installed and the pip package manager.

Before running the project, you'll need to have, 

- OpenAI API key (purchage/get one from here: https://platform.openai.com/). Make sure to set this up, as it is a crucial part of the project.
- Tavily search API (get a free one or purchase here: https://app.tavily.com/home)

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

- For a quick and dirty intro follow the sample jupyter notebook, Experiments.ipynb, given in the repo.

## License

This software is licensed under the GNU General Public License (GPL), version 3 (GPL-3.0). For the full text of the license, see the [LICENSE](LICENSE) file in this repository.

For more information on GPL-3.0, visit: https://www.gnu.org/licenses/gpl-3.0.en.html

## Citation

If you use this for any evaluation or wrote any article that levrages this tool, please cite me: https://arxiv.org/abs/2405.16072
Here is the bibtext version for your convenience:

```
@misc{sheikholeslam2024synthai,
      title={SynthAI: A Multi Agent Generative AI Framework for Automated Modular HLS Design Generation}, 
      author={Seyed Arash Sheikholeslam and Andre Ivanov},
      year={2024},
      eprint={2405.16072},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

## Status

Please note that this project is currently under active development. Feedback and contributions are always welcome!
