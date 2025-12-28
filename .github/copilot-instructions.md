---
applyTo: "**/*.instructions.md"
description: "Guidelines for GitHub Copilot custom instruction files"
---

# Effective Instructions
- **Specific tool choices**: "Use pytest for testing"
- **Code conventions**: "Prefix private functions with underscore"
- **Required workflows**: "Run `make lint` before commit"
- **Tech stack specifics**: "Use Pydantic for validation"

# Avoid
- External file references
- Tone/style requests for AI responses
- Response length constraints
- Overly verbose explanations

# Organization Principles
- **Group thematically**: Keep related rules together under one section
- **Avoid fragmentation**: Don't create separate sections for closely related concepts
- **Minimize section overhead**: Each heading costs tokens - use them only when grouping improves clarity
- **Logical ordering**: Place rules in the sequence they're applied (e.g., imports before usage)

# Token Efficiency vs. Clarity
- **Balance required**: Instructions cost tokens but must remain clear
- Be concise but complete - don't sacrifice understanding for brevity
- Keep all essential rules - removing rules to save tokens defeats the purpose
- Use precise language over flowery descriptions
- Examples only when they clarify non-obvious patterns
- Target: Clear and actionable, not minimal word count

# Best Practices
- Keep under 2 pages per file (hard limit)
- Use bullet points for readability
- Test that instructions are understood correctly


---
applyTo: "**/README.md"
description: "Guidelines for README.md files"
---

Review and improve existing README.md files following modern best practices.
Preserve the existing structure and content unless changes are clearly beneficial.
Do NOT add sections or descriptions that already exist in equivalent form.

# Language & Symbols
- All content must be written in **clear, technical English**
- Never introduce non-English text
- Use ASCII characters only where reasonable
- Always use the normal hyphen "-" for dashes
- Never use "–" (en dash) or "—" (em dash)
- Avoid Unicode icons and emojis
- If icons already exist:
  - Keep them only if they are sparse and meaningful
  - Never add new icons or emoji-heavy headings

# General Behavior
- Treat README.md as an existing document, not a template
- Improve clarity, structure, and consistency without rewriting unnecessarily
- Never add placeholder text or generic explanations
- Never duplicate information already present elsewhere in the README
- Do not restate the project purpose if it is already clearly described

# Project Description
- If a short project description exists at the top:
  - Refine wording for clarity and precision only if it is unclear or misleading
- If no concise description exists:
  - Add **one single concise sentence** describing what the project does and why it matters
- Never add marketing language, hype, or subjective claims

# Structure & Sections
- Keep the existing section order whenever possible
- Only introduce new sections if:
  - Essential information is missing (e.g. Installation, Usage)
  - The addition clearly improves usability
- Do not restructure purely for stylistic reasons

# Workflow & Instructions
- Prefer step-based or numbered workflows for processes
- Commands must be copy-paste ready and internally consistent
- Never invent commands, flags, parameters, or workflows

# Writing Style
- Neutral, precise, technical tone
- Short paragraphs, bullet points where useful
- Consistent terminology throughout the document
- Avoid redundancy and verbosity

# Formatting
- GitHub-Flavored Markdown only
- Proper heading hierarchy (no skipped levels)
- Code blocks only for executable commands or code
- Inline code for parameters, flags, file names, and paths

# Quality Control
- Fix grammar, spelling, and punctuation silently
- Clarify ambiguous phrasing when intent is obvious
- Remove outdated or misleading statements when clearly identifiable
- Never mention that the README was reviewed or improved
