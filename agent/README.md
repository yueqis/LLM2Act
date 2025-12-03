# VirtualHome Agent

LLM-powered agent for VirtualHome simulator with tool generation and reuse.

## Features

- **Tool Reuse**: Built-in skills (walk_to, grab_object, put_on, turn_on, sit_on, etc.)
- **Tool Generation**: LLM can define new reusable tools on-the-fly
- **Script Validation**: Validates action sequences before execution
- **Video Recording**: Automatically generates MP4 videos of task execution

## Setup

1. Clone VirtualHome API:
```bash
git clone https://github.com/xavierpuigf/virtualhome.git
```

2. Download VirtualHome executable from [releases](https://github.com/xavierpuigf/virtualhome/releases)

3. Install dependencies:
```bash
pip install openai opencv-python ipdb ffmpeg-python
```

## Run

```bash
export OPENAI_API_KEY=your-api-key
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4o
export VH_EXECUTABLE=/path/to/macos_exec.2.2.4.app
export PYTHONPATH=/path/to/virtualhome/virtualhome:$PYTHONPATH

# Run a single task
python agent.py --task "Sit on the sofa"

# Interactive mode
python agent.py

# Run examples
python agent.py --examples

# Offline mode (no simulator)
python agent.py --offline --task "grab an apple"
```

## Commands (Interactive Mode)

- `reset` - Reset the environment
- `tools` - List available tools
- `quit` - Exit

## Output

- Video saved to `Output/task/0/task_video.mp4`
- Use `--no-record` to disable video recording

## Architecture

```
agent.py          # Main entry point
planner.py        # Planning with tool reuse
tool_generator.py # Tool/Skill definitions and LLM generation
action_mapper.py  # VirtualHome action definitions
```
