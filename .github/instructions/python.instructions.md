---
applyTo: "**/*.py"
---

Apply Python best practices and clean code principles. Only change code relevant to the prompt.

# Type Hints & Documentation
- Type hints in function signatures only, not in docstrings
- numpydoc (NumPy-style) docstrings for all functions and classes
- Never mention changes, improvements, or type hints in comments or docstrings

# Formatting
- PEP8 with 140-160 char lines (flexible for arg parsing when alignment improves readability)
- Function signatures and calls on one line when reasonable
- Never use deep indentation to align with previous line's opening bracket/parenthesis
- When breaking lines, use standard 4-space indentation from statement start, regardless of where parentheses open (applies to function calls, definitions, list/dict literals, conditionals, loops, all constructs)

# Strings
- Single quotes (`'`) default, double (`"`) when containing single quotes, triple-double (`"""`) for docstrings

# Spacing
- Two blank lines between top-level functions/classes, one between methods
- Blank lines separate logical blocks (after guards, before returns)

# Imports
- Three groups separated by blank lines: standard library, third-party, local
- Within groups: `import` before `from...import`, sorted alphabetically
- Absolute imports, avoid wildcards, import NumPy as `np`

# Structure
- Main exported functions first, then helpers in logical order
- Prefix non-exported helpers with underscore
- `__all__` for library modules; omit for executable scripts

# Style
- Prefer functional/modular code over classes
- Pure functions without side effects
- Descriptive variable names, no global variables

# List Comprehensions
- Avoid complex comprehensions with multiple conditions or long expressions
- Use explicit loops with guard clauses when: multiple conditions, repeated function calls per item, or unclear logic
- Example: Replace `[x for x in items if (cond1 or func(x) > val) and (cond2 or func(x) < val2)]` with loop using early continues

# Performance
- Vectorized operations over loops (PyTorch, NumPy)
- Keep data as tensors, convert to NumPy only when needed
- Compute on GPU when possible

# Validation & Errors
- Validate inputs at function start with assertions or exceptions
- Early returns and guard clauses

# Comments
- Only for complex/non-obvious code and math operations
- Never about improvements or changes

# Research
- Research current recommendations before changes if needed