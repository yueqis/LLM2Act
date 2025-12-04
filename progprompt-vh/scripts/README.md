# Agent Evaluation Scripts

This directory contains scripts for evaluating the VirtualHome Agent on ProgPrompt-VH benchmark tasks.

## Quick Start

```bash
cd /workspace/project/LLM2Act/progprompt-vh/scripts

# Set environment
export DISPLAY=:99
export PYTHONPATH=/workspace/project/LLM2Act/virtualhome/virtualhome:$PYTHONPATH

# Run evaluation
python run_agent_eval.py \
  --progprompt-path .. \
  --expt-name my_agent_eval \
  --unity-filename /workspace/project/LLM2Act/virtualhome/linux_exec.v2.3.0.x86_64 \
  --test-set test_seen \
  --agent-model gpt-4o \
  --init-basic-skills \
  --port 8093 \
  --display 99
```

## Scripts

### `run_agent_eval.py`
Main evaluation script that runs the agent on VirtualHome tasks and evaluates performance.

**Key Features**:
- Generates plans for all tasks in test set
- Executes plans in VirtualHome simulator
- Evaluates results using ProgPrompt-VH metrics
- Saves plans, tool registry, and detailed logs

### `agent_adapter.py`
Adapter class that bridges the agent implementation and progprompt-vh framework.

**Key Features**:
- Initializes agent with progprompt-vh's Unity connection
- Converts between VirtualHome and pythonic formats
- Manages tool registry and metadata

### `test_agent_integration.py`
Quick test script to verify the integration works correctly.

```bash
python test_agent_integration.py
```

## Output Files

Results are saved in `../results/` with pattern:
- `{expt_name}_agent_{model}_{test_set}_with_basic_skills_plans.json`
- `{expt_name}_agent_{model}_{test_set}_with_basic_skills_tool_registry.json`
- `{expt_name}_agent_{model}_{test_set}_with_basic_skills_logs.txt`
- `{expt_name}_agent_{model}_{test_set}_with_basic_skills_metrics.json`

## Documentation

See `../AGENT_INTEGRATION.md` for comprehensive documentation including:
- Architecture details
- All command-line arguments
- Examples and use cases
- Evaluation metrics explanation
- Troubleshooting guide

## Baseline Comparison

To compare with baseline ProgPrompt:

```bash
# Run agent evaluation
python run_agent_eval.py --progprompt-path .. --expt-name agent \
  --unity-filename /path/to/simulator --test-set test_seen \
  --agent-model gpt-4o --init-basic-skills --port 8093 --display 99

# Run baseline ProgPrompt
python run_eval.py --progprompt-path $(pwd) --expt-name progprompt \
  --openai-api-key $(cat ../../api_key.txt) \
  --unity-filename /path/to/simulator \
  --test-set test_seen --prompt-num-examples 3

# Compare results in ../results/
```

## Key Parameters

- `--agent-model`: OpenAI model (default: gpt-4o)
- `--init-basic-skills`: Start with foundational tools (recommended)
- `--test-set`: test_seen, test_unseen, test_unseen_ambiguous, env1, env2
- `--load-generated-plans`: Skip plan generation, use cached plans
- `--port`: Unity port (use different port for each run to avoid conflicts)
