# FunnyBhatiaji

## Project Description
**FunnyBhatiaji** is a fictional Indian stand-up comedian designed to introduce himself in a humorous, India-inspired way. He can retain the context of previous jokes, using it cleverly in his next joke to maintain a seamless flow.

## Changes Made
### 1. `main.py`
- Added a `joke` variable to store the previous joke.
  - If **FunnyBhatiaji** is the first to perform, he introduces himself with a unique, introductory joke.
  - Otherwise, he builds on the existing joke context, adding his spin to it.

### 2. `api.py`
- Applied the same approach as in `main.py`.
  - Added an option to pass the previous joke to the `tell_joke` method.
  - When a previous joke is provided, it serves as the context for creating a new joke.
  - If no argument is passed, an introductory joke is constructed.

## API Integration
- Integrated the Hugging Face Inference Client API to leverage the **Llama model** for joke generation.
- Using this cloud API ensures hardware compatibility, as server specifications are unknown.
