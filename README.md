# LangGraph Demo Project

This project demonstrates the use of the LangGraph library to create a structured chat agent using the Anthropic Claude model. The agent can generate Python code based on user requests.

## Prerequisites

- Python 3.7 or higher
- `pip` (Python package installer)
- An Anthropic API key

## Installation

1. **Clone the repository**:
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment** (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Create a `.env` file in the root directory of the project.
   - Add your Anthropic API key to the `.env` file:
     ```plaintext
     ANTHROPIC_API_KEY=your_anthropic_api_key
     ```

## Usage

1. **Run the script**:
   ```sh
   python demo-langgraph.py
   ```

2. **Example request**:
   - The script will generate Python code based on a request. You can modify the `feature_request` variable in the script to change the request.

## Functions

### `create_claude_llm()`
Creates and returns an instance of the Anthropic Claude model.

### `write_code(request: str) -> str`
Generates Python code based on the provided request using the Claude model.

## Logging

The project uses Python's built-in logging module to log information and errors. Logs are set to the ERROR level by default.

## License

This project is licensed under the MIT License. See the LICENSE file for details.