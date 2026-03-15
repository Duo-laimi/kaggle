# AGENTS.md - Developer Guidelines for AIMO3 Math Project

## Project Overview

This is a Kaggle AIMO3 math competition project that uses LLMs to solve math problems. The project supports both local models (via transformers) and online API models (via OpenAI-compatible APIs).

## Build, Test & Development Commands

### Running Tests

Run a single test file directly with Python:
```bash
python test/test_ccm.py
python test/test_solver.py
python test/test_json.py
```

Run all tests in the test directory:
```bash
python -m pytest test/  # if pytest is installed
```

### Running the Application

```bash
# Main entry point
python main.py

# Using configuration
python -c "from model.solver import KaggleSolver; ..."
```

### Development

No formal build system (Makefile, setup.py) currently exists. All code runs directly from source.

## Code Style Guidelines

### Imports

- Use absolute imports for project modules: `from model.utils.stream import collect_api_stream`
- Use relative imports within packages: `from .format import extract_think`
- Group imports in order: standard library, third-party, local
- Avoid wildcard imports: `from module import *`

Example:
```python
import json
import re
from typing import Any, Dict, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.solver import KaggleSolver
from model.utils.stream import collect_model_stream
```

### Formatting

- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters (soft guideline)
- Use blank lines to separate logical sections
- No trailing whitespace

### Types

- Use type hints for function parameters and return values
- Use `typing` module for complex types
- Avoid `Any` when possible

Good:
```python
def extract_answer(text: str) -> Optional[int]:
    ...
```

### Naming Conventions

- **Variables/Functions**: snake_case (`collect_model_stream`, `max_length`)
- **Classes**: PascalCase (`KaggleSolver`, `AIMO3Sandbox`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_TURNS`, `DEFAULT_TIMEOUT`)
- **Private members**: prefix with underscore (`_private_method`)

### Error Handling

- Use specific exception types
- Include meaningful error messages
- Handle exceptions at appropriate levels

### Code Patterns

1. **Stream Processing** (see `model/utils/stream.py`):
   - Generator functions for streaming output
   - Separate collection phase from processing
   - Return tuples for multiple outputs

2. **Configuration** (see `config/*.yaml`):
   - Use YAML files for configuration
   - Load with `yaml.safe_load()`
   - Pass config dicts to classes

3. **Tools** (see `model/tool/base.py`, `model/tool/sandbox.py`):
   - Implement tool interface with `name` and schema
   - Tools return string results for LLM consumption

### Project Structure

```
aimo3_math/
├── model/                  # Main model code
│   ├── solver.py          # KaggleSolver class
│   ├── model.py           # Model loading utilities
│   ├── tool/              # Tool implementations
│   │   ├── base.py
│   │   └── sandbox.py
│   └── utils/             # Utility functions
│       ├── format.py      # Text extraction/parsing
│       ├── prompt.py      # Prompt templates
│       └── stream.py      # Stream collection
├── test/                  # Test files
├── config/                # YAML configurations
├── datapre/              # Data preprocessing
└── utils/                # General utilities
```

### Key Conventions

1. **Chat Templates**: Use `tokenizer.apply_chat_template()` for conversation formatting
2. **Thinking**: Models use `<think>` / `</think>` tokens for reasoning
3. **Tool Calls**: Use `<tool_call>` / `</tool_call>` XML tags
4. **Answers**: Wrap final answers in `\boxed{...}` LaTeX format
5. **Message Format**: Follow OpenAI's ChatCompletionMessage structure

### Testing Guidelines

- Test files in `test/` directory
- Name test files: `test_<module>.py`
- Use direct Python execution for simple tests
- Include verbose output for debugging stream processing

Example test pattern:
```python
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.utils.stream import collect_model_stream

def test_collect_model_stream():
    model_output = "..."
    
    def text_stream():
        for chunk in model_output.split():
            yield chunk
    
    content, reasoning_content, tool_calls = collect_model_stream(text_stream())
    assert content is not None
    
    print("All tests passed!")

if __name__ == "__main__":
    test_collect_model_stream()
```

### Common Patterns

1. **Model Inference** (offline mode):
   - Load model with `AutoModelForCausalLM.from_pretrained()`
   - Apply chat template with `tokenizer.apply_chat_template()`
   - Generate with `model.generate(**inputs, max_length=...)`

2. **API Inference** (online mode):
   - Use OpenAI client with base_url and api_key
   - Call `client.chat.completions.create()` with `stream=True`
   - Collect stream with `collect_api_stream()`

3. **Offline Mode Detection**:
   ```python
   self.offline_mode = True  # default
   if base_url is not None:
       self.offline_mode = False
       self.client = OpenAI(base_url=base_url, api_key=api_key)
   ```

### LLM/AI Specific Guidelines

- When adding new stream processing, maintain compatibility with both API and local model streams
- Consider thinking content (`reasoning_content`) for models that support it
- Handle tool calls consistently between online and offline modes
- Use verbose flags for debugging streaming output
