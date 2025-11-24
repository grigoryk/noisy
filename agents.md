# Agent Corrections Log

## 2025-11-22

### Code Comments
- User prefers minimal comments
- Only comment non-obvious code
- Remove docstrings and routine explanations

### Code Style
- Keep code modular and flexible
- Very minimalist approach
- Split functionality into clear, focused functions
- Use descriptive function names to show what each function does
- Keep functions small and readable
- Organize related code together for logical grouping
- Make it easy to see what's responsible for what

### Context Management
- At the start of each session, read entire codebase to build comprehensive understanding
- Keep detailed mental model of the code structure in context
- Maintain awareness of what's what to avoid redundant lookups
- Understand relationships between components and data flow
- Look things up when needed or confused - that's fine

### Agent Behavior
- Always update agents.md file when user gives new instructions or preferences
- Keep the log current with latest guidance

### Session Context Management
- At the start of each session, read session_context.md to understand project state
- Update session_context.md at the end of significant work sessions with:
  - New features added
  - Architecture changes
  - Configuration variable additions
  - Known issues or observations about current implementation
  - Recent changes summary
- If user asks to "save session context" or at natural stopping points, update the file
- session_context.md serves as project memory across sessions
- Keep it factual and technical - document what exists and observations about current behavior
- Observations section can note limitations or characteristics of current implementation

### Performance Profiling Approach
- Use line_profiler for performance analysis (not cProfile first)
- Start by analyzing where the program actually spends time
- Begin at the top-level entry point (e.g., main(), run())
- Add @line_profiler.profile decorators to functions where significant time is spent
- Work top-down: profile entry point first, then drill into hot functions
- This shows line-by-line timing to identify actual bottlenecks, not just function call counts

## 2025-11-23

### Profiling Workflow
- Toggle profiling by exporting `LINE_PROFILE=1` (on) or `LINE_PROFILE=0` (off) before running `python main.py`
- Inspect captured stats with `python -m line_profiler -rtmz profile_output.lprof`
- Use `@line_profiler.profile` on functions worth instrumenting
- Each session is limited to 20 profiling executions, so track attempts carefully

## 2025-11-24

### Profiling Iteration Checklist
- Always profile the actual entry path (`AudioVisualizer.run` via `main.py`) before drilling into helpers.
- For each run: export `LINE_PROFILE=1`, then launch `env/bin/python main.py --run-for <seconds>` to capture a short, repeatable clip (keep it under ~30s to save executions).
- After each profiled run, dump stats with `python -m line_profiler -rtmz profile_output.lprof` and archive the human-readable text (e.g., `profile_output_YYYY-MM-DDTHHMMSS.txt`).
- Only add new `@line_profiler.profile` decorators to functions you plan to inspect next; remove or collapse them after addressing bottlenecks to keep noise low.
- Stay mindful of the 20-run cap per session—log each attempt and stop early if findings are clear.
- After wrapping a profiling loop, provide an impact summary that lists the changes kept, the % impact for each improvement, and the cumulative % improvement so far—skip discussing “what remains” (save that for the next loop).
