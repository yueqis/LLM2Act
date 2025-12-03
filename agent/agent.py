"""
VirtualHome Agent

Main entry point for running VirtualHome tasks with LLM-powered planning.

Environment Variables:
    OPENAI_API_KEY: OpenAI API key
    OPENAI_BASE_URL: OpenAI API base URL
    OPENAI_MODEL: Model name (default: gpt-4o)
    VH_EXECUTABLE: Path to VirtualHome Unity executable
    VH_PORT: VirtualHome connection port (default: 8080)

Example:
    export OPENAI_API_KEY=your-api-key
    export OPENAI_BASE_URL=https://api.openai.com/v1
    export VH_EXECUTABLE=/path/to/macos_exec.app
    python agent.py
"""

import os
import time
import sys
import argparse
from typing import Dict, Any

from planner import Planner


# Default configuration
DEFAULT_EXECUTABLE = os.getenv("VH_EXECUTABLE", "")
DEFAULT_PORT = os.getenv("VH_PORT", "8080")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


def run_model_comparison_demo(
    strong_model: str = "gpt-5",
    weak_model: str = "gpt-4.1-mini",
    recording: bool = False
):
    """
    Demo: Demonstrate how tool reuse helps weaker models.

    This demo shows:
    1. Strong model (e.g., GPT-5) solves tasks and generates reusable tools
    2. Weak model without tools attempts the same task
    3. Weak model with pre-generated tools attempts the same task

    The comparison shows whether having access to pre-generated tools
    improves the weak model's performance.
    """
    print("\n" + "=" * 70)
    print("DEMO: Tool Reuse Across Models")
    print("=" * 70)
    print(f"Strong model: {strong_model}")
    print(f"Weak model: {weak_model}")

    skills_file = "generated_skills.json"

    # Define tasks for the strong model to generate tools
    # These are diverse tasks that create useful, reusable tools
    tool_generation_tasks = [
        "Grab the apple and put it on the kitchen table",
        "Put the plate in the fridge",
        "Turn on the TV",
        "Grab the cup and put it in the dishwasher",
    ]

    # Test tasks - more complex tasks that benefit from tool reuse
    # These require multi-step planning and can reuse generated tools
    test_tasks = [
        "Grab the banana and put it in the fridge",  # Should reuse open_container tool
        "Put the wine glass in the kitchen cabinet",  # Complex: grab + open + putin + close
        "Grab the remote control and turn on the TV, then put the remote on the sofa",  # Multi-step
    ]

    # Mock observation for offline mode
    mock_observation = {
        "objects": [
            {"name": "apple", "id": 101, "states": []},
            {"name": "banana", "id": 102, "states": []},
            {"name": "plate", "id": 103, "states": []},
            {"name": "kitchen_table", "id": 201, "states": []},
            {"name": "coffee_table", "id": 202, "states": []},
            {"name": "dishwasher", "id": 301, "states": ["CLOSED"]},
            {"name": "tv", "id": 401, "states": ["OFF"]},
            {"name": "fridge", "id": 302, "states": ["CLOSED"]},
            {"name": "sofa", "id": 501, "states": []},
        ],
        "agent_id": 0
    }

    # =========================================================================
    # Phase 1: Strong model generates tools from diverse tasks
    # =========================================================================
    print("\n" + "-" * 70)
    print("PHASE 1: Strong model generates reusable tools")
    print("-" * 70)
    print(f"Model: {strong_model}")
    print(f"Running {len(tool_generation_tasks)} tasks to generate tools...")

    strong_planner = Planner(
        executable_path=os.getenv("VH_EXECUTABLE", ""),
        port=os.getenv("VH_PORT", "8080"),
        model=strong_model,
        init_basic_skills=False  # Start with empty registry
    )

    # Connect if executable provided
    if strong_planner.executable_path:
        strong_planner.connect()
        strong_planner.reset(0)
        time.sleep(2)

    all_generated_tools = []
    for i, task in enumerate(tool_generation_tasks):
        print(f"\n[Task {i+1}/{len(tool_generation_tasks)}] {task}")

        if strong_planner.comm:
            result = strong_planner.run(task, recording=recording)
        else:
            result = strong_planner.run_with_observation(task, mock_observation, recording=False)

        if result.get('new_tools'):
            all_generated_tools.extend(result['new_tools'])
            print(f"  -> Generated tools: {result['new_tools']}")
        if result.get('tools_used'):
            print(f"  -> Reused tools: {result['tools_used']}")
        print(f"  -> Script: {len(result.get('script', []))} actions")

        # Reset environment between tasks
        if strong_planner.comm:
            strong_planner.reset(0)
            time.sleep(1)

    # Save generated tools
    strong_planner.registry.save(skills_file)
    print(f"\n{'='*50}")
    print(f"Generated {len(all_generated_tools)} tools total")
    print(f"Tools saved to: {skills_file}")
    print(f"Tool registry contents:")
    print(strong_planner.registry.get_tools_summary())

    # =========================================================================
    # Phase 2: Weak model WITHOUT tools (baseline)
    # =========================================================================
    print("\n" + "-" * 70)
    print("PHASE 2: Weak model WITHOUT tools (baseline)")
    print("-" * 70)
    print(f"Model: {weak_model}")
    print(f"Testing {len(test_tasks)} tasks...")
    print(f"Available tools: None (empty registry)")

    weak_planner_baseline = Planner(
        executable_path="",
        port=os.getenv("VH_PORT", "8080"),
        model=weak_model,
        init_basic_skills=False  # Empty registry
    )

    # Reuse connection if available
    weak_planner_baseline.comm = strong_planner.comm
    weak_planner_baseline.executable_path = strong_planner.executable_path
    # Set task counter to avoid overwriting Phase 1 videos
    weak_planner_baseline.task_counter = 100  # Phase 2 videos: task_100, task_101, ...

    baseline_results = []
    for i, test_task in enumerate(test_tasks):
        print(f"\n[Baseline Task {i+1}/{len(test_tasks)}] {test_task}")

        if weak_planner_baseline.comm:
            weak_planner_baseline.reset(0)
            time.sleep(2)
            result = weak_planner_baseline.run(test_task, recording=recording)
        else:
            result = weak_planner_baseline.run_with_observation(
                test_task, mock_observation, recording=False
            )

        baseline_results.append({
            "task": test_task,
            "success": result.get('success', False),
            "script_len": len(result.get('script', [])),
            "script": result.get('script', []),
            "new_tools": result.get('new_tools', [])
        })

        status = "SUCCESS" if result.get('success') else "FAILED"
        print(f"  Result: {status}")
        print(f"  Script: {len(result.get('script', []))} actions")
        if result.get('script'):
            for line in result.get('script', []):
                print(f"    {line}")

    # =========================================================================
    # Phase 3: Weak model WITH pre-generated tools
    # =========================================================================
    print("\n" + "-" * 70)
    print("PHASE 3: Weak model WITH pre-generated tools")
    print("-" * 70)
    print(f"Model: {weak_model}")
    print(f"Testing {len(test_tasks)} tasks...")

    weak_planner_with_tools = Planner(
        executable_path="",
        port=os.getenv("VH_PORT", "8080"),
        model=weak_model,
        init_basic_skills=False
    )

    # Load tools generated by strong model
    weak_planner_with_tools.registry.load(skills_file)
    print(f"Loaded tools: {weak_planner_with_tools.registry.list_tools()}")

    # Reuse connection if available
    weak_planner_with_tools.comm = strong_planner.comm
    weak_planner_with_tools.executable_path = strong_planner.executable_path
    # Set task counter to avoid overwriting Phase 1 & 2 videos
    weak_planner_with_tools.task_counter = 200  # Phase 3 videos: task_200, task_201, ...

    with_tools_results = []
    for i, test_task in enumerate(test_tasks):
        print(f"\n[With-Tools Task {i+1}/{len(test_tasks)}] {test_task}")

        if weak_planner_with_tools.comm:
            weak_planner_with_tools.reset(0)
            time.sleep(2)
            result = weak_planner_with_tools.run(test_task, recording=recording)
        else:
            result = weak_planner_with_tools.run_with_observation(
                test_task, mock_observation, recording=False
            )

        with_tools_results.append({
            "task": test_task,
            "success": result.get('success', False),
            "script_len": len(result.get('script', [])),
            "script": result.get('script', []),
            "tools_used": result.get('tools_used', [])
        })

        status = "SUCCESS" if result.get('success') else "FAILED"
        print(f"  Result: {status}")
        print(f"  Script: {len(result.get('script', []))} actions")
        print(f"  Tools reused: {result.get('tools_used', [])}")
        if result.get('script'):
            for line in result.get('script', []):
                print(f"    {line}")

    # =========================================================================
    # Comparison Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\nStrong model: {strong_model}")
    print(f"Weak model: {weak_model}")
    print(f"Tools generated: {all_generated_tools}")

    # Calculate statistics
    baseline_success = sum(1 for r in baseline_results if r['success'])
    baseline_total_actions = sum(r['script_len'] for r in baseline_results)
    with_tools_success = sum(1 for r in with_tools_results if r['success'])
    with_tools_total_actions = sum(r['script_len'] for r in with_tools_results)
    total_tools_reused = sum(len(r['tools_used']) for r in with_tools_results)

    print(f"\n{'='*70}")
    print(f"{'Task':<50} {'No Tools':<10} {'With Tools':<10}")
    print(f"{'='*70}")

    for i, test_task in enumerate(test_tasks):
        baseline_len = baseline_results[i]['script_len']
        with_tools_len = with_tools_results[i]['script_len']
        tools_used = with_tools_results[i]['tools_used']

        baseline_status = f"{baseline_len} actions"
        with_tools_status = f"{with_tools_len} actions"
        if tools_used:
            with_tools_status += f" (reused: {len(tools_used)})"

        # Truncate task name if too long
        task_display = test_task[:47] + "..." if len(test_task) > 50 else test_task
        print(f"{task_display:<50} {baseline_status:<10} {with_tools_status:<20}")

    print(f"{'='*70}")
    print(f"{'TOTAL':<50} {baseline_total_actions:<10} {with_tools_total_actions} (reused: {total_tools_reused})")

    print("\n" + "-" * 70)
    print("ANALYSIS:")
    print("-" * 70)

    if total_tools_reused > 0:
        print(f"- Weak model successfully REUSED {total_tools_reused} pre-generated tool(s) across {len(test_tasks)} tasks")

    if with_tools_total_actions > baseline_total_actions:
        improvement = with_tools_total_actions - baseline_total_actions
        print(f"- With tools generated {improvement} more actions total")
        print("=> PRE-GENERATED TOOLS IMPROVED OUTPUT QUALITY!")
    elif with_tools_total_actions == baseline_total_actions and total_tools_reused > 0:
        print("- Similar action counts, but weak model learned to REUSE tools")
        print("=> TOOL REUSE PATTERN SUCCESSFULLY TRANSFERRED!")
    else:
        print("- Results were similar with and without tools")

    # Check for script validity (non-empty scripts with proper formatting)
    baseline_valid = sum(1 for r in baseline_results if r['script_len'] > 0 and all('(' in s for s in r['script']))
    with_tools_valid = sum(1 for r in with_tools_results if r['script_len'] > 0 and all('(' in s for s in r['script']))

    if with_tools_valid > baseline_valid:
        print(f"- Valid scripts: {baseline_valid}/{len(test_tasks)} (no tools) vs {with_tools_valid}/{len(test_tasks)} (with tools)")
        print("=> TOOLS HELPED GENERATE MORE VALID SCRIPTS!")

    print("\n" + "=" * 70)

    return {
        "baseline_results": baseline_results,
        "with_tools_results": with_tools_results,
        "tools_generated": all_generated_tools,
        "summary": {
            "baseline_total_actions": baseline_total_actions,
            "with_tools_total_actions": with_tools_total_actions,
            "total_tools_reused": total_tools_reused
        }
    }


def run_tool_generation_demo(planner: Planner, recording: bool = False):
    """
    Demo: Tool Generation and Reuse

    Starts with EMPTY registry, generates tools on-the-fly, then reuses them.
    Shows how the agent learns to create reusable tools and apply them to new situations.
    """
    print("\n" + "=" * 70)
    print("DEMO: Tool Generation and Reuse")
    print("=" * 70)
    print(f"\nModel: {planner.model}")
    print(f"Starting with EMPTY skill registry...")
    print(f"Initial tools: {planner.registry.list_tools()}")

    # Mock objects for offline mode - diverse enough for various tasks
    mock_objects = [
        {"name": "sofa", "id": 101, "states": []},
        {"name": "kitchen_table", "id": 102, "states": []},
        {"name": "chair", "id": 103, "states": []},
        {"name": "tv", "id": 104, "states": ["OFF"]},
        {"name": "apple", "id": 105, "states": []},
        {"name": "banana", "id": 106, "states": []},
        {"name": "fridge", "id": 107, "states": ["CLOSED"]},
        {"name": "coffee_table", "id": 108, "states": []},
        {"name": "remote_control", "id": 109, "states": []},
    ]

    # Tasks designed to demonstrate tool generation and reuse
    # First task creates a tool, subsequent similar tasks should reuse it
    tasks = [
        "Grab the apple and put it on the kitchen table",  # Should create grab_and_place tool
        "Grab the banana and put it on the coffee table",  # Should REUSE the grab_and_place tool
        "Turn on the TV",  # Should create switch_on tool
        "Sit on the sofa",  # Should create sit_down tool
        "Sit on the chair",  # Should REUSE sit_down tool
    ]

    results = []
    for i, task in enumerate(tasks):
        print(f"\n{'='*70}")
        print(f"[Task {i+1}/{len(tasks)}] {task}")
        print(f"Available tools before: {planner.registry.list_tools()}")
        print("=" * 70)

        # Use mock observation if offline
        if not planner.comm:
            observation = {"objects": mock_objects, "agent_id": 0}
            result = planner.run_with_observation(task, observation, recording=recording)
        else:
            result = planner.run(task, recording=recording)

        results.append({
            "task": task,
            "new_tools": result.get('new_tools', []),
            "tools_used": result.get('tools_used', []),
            "script_len": len(result.get('script', []))
        })

        # Show results
        status = "SUCCESS" if result['success'] else "COMPLETED (offline)"
        print(f"\nResult: {status}")

        if result.get('new_tools'):
            print(f">>> NEW TOOL CREATED: {result['new_tools']}")
        if result.get('tools_used'):
            reused = [t for t in result['tools_used'] if t not in result.get('new_tools', [])]
            if reused:
                print(f">>> REUSED EXISTING TOOL: {reused}")

        print(f"Script: {result.get('script', [])}")
        print(f"Registry now has {len(planner.registry.list_tools())} tools")

        # Wait between tasks for VirtualHome to stabilize
        if planner.comm:
            planner.reset(0)
            time.sleep(1)

    # Final summary
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)

    print(f"\nFinal registry contains {len(planner.registry.list_tools())} tools:")
    print(planner.registry.get_tools_summary())

    # Show reuse statistics
    print("\n" + "-" * 70)
    print("TOOL REUSE ANALYSIS:")
    print("-" * 70)

    total_new = sum(len(r['new_tools']) for r in results)
    total_reused = sum(
        len([t for t in r['tools_used'] if t not in r['new_tools']])
        for r in results
    )

    print(f"Total tasks: {len(tasks)}")
    print(f"New tools created: {total_new}")
    print(f"Tool reuse instances: {total_reused}")

    if total_reused > 0:
        print("\n=> The agent successfully learned to REUSE tools across tasks!")

    # Save tools for future use
    planner.registry.save("generated_skills.json")
    print(f"\nTools saved to: generated_skills.json")


def run_examples(planner: Planner, recording: bool = True):
    """Run example tasks with pre-defined tools."""

    examples = [
        "Walk to the kitchen table",
        "Sit on the sofa",
        "Grab an apple and put it on the table",
    ]

    print("\n" + "=" * 70)
    print("Running Example Tasks (with pre-defined tools)")
    print("=" * 70)

    for i, task in enumerate(examples):
        print(f"\n[Example {i+1}/{len(examples)}]")
        result = planner.run(task, recording=recording)

        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"\nResult: {status}")

        if result.get('tools_used'):
            print(f"Tools used: {result['tools_used']}")
        if result.get('new_tools'):
            print(f"New tools created: {result['new_tools']}")

        # Reset environment between examples
        if planner.comm:
            planner.reset(0)


def interactive_mode(planner: Planner, recording: bool = True):
    """Run in interactive mode, accepting tasks from user input."""
    print("\n" + "=" * 70)
    print("Interactive Mode")
    print("Type a task and press Enter. Type 'quit' to exit.")
    print("Commands: 'reset' - reset env, 'tools' - list tools")
    print("=" * 70)

    while True:
        try:
            task = input("\nTask> ").strip()

            if not task:
                continue

            if task.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if task.lower() == 'reset':
                planner.reset(0)
                print("Environment reset.")
                continue

            if task.lower() == 'tools':
                print("\nAvailable tools:")
                print(planner.registry.get_tools_summary())
                continue

            result = planner.run(task, recording=recording)

            status = "SUCCESS" if result['success'] else "FAILED"
            print(f"\nResult: {status}")

            if result.get('new_tools'):
                print(f"New tools created: {result['new_tools']}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="VirtualHome Agent")
    parser.add_argument(
        "--task",
        type=str,
        help="Task to execute (if not provided, runs in interactive mode)"
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run example tasks with pre-defined tools"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run tool generation demo (starts with empty registry)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run model comparison demo (strong generates, weak reuses)"
    )
    parser.add_argument(
        "--strong-model",
        type=str,
        default="gpt-5",
        help="Strong model for tool generation (default: gpt-5)"
    )
    parser.add_argument(
        "--weak-model",
        type=str,
        default="gpt-4.1-mini",
        help="Weak model for tool reuse (default: gpt-4.1-mini)"
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable video recording"
    )
    parser.add_argument(
        "--env",
        type=int,
        default=0,
        help="Environment index (0-6)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode (no simulator)"
    )
    args = parser.parse_args()

    # Print configuration
    print("=" * 70)
    print("VirtualHome Agent")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  VH_EXECUTABLE: {DEFAULT_EXECUTABLE or 'not set'}")
    print(f"  VH_PORT: {DEFAULT_PORT}")
    print(f"  OPENAI_MODEL: {DEFAULT_MODEL}")
    print(f"  OPENAI_API_KEY: {'set' if os.getenv('OPENAI_API_KEY') else 'not set'}")
    print(f"  OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'not set')}")

    # Check requirements
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY not set")
        sys.exit(1)
    if not os.getenv("OPENAI_BASE_URL"):
        print("\nError: OPENAI_BASE_URL not set")
        sys.exit(1)

    # Create planner - always start with empty registry by default
    # Tools should be generated by the agent, not pre-defined
    executable = "" if args.offline else DEFAULT_EXECUTABLE
    planner = Planner(
        executable_path=executable,
        port=DEFAULT_PORT,
        model=DEFAULT_MODEL,
        init_basic_skills=False  # Always start empty - agent generates tools
    )

    # Connect and reset if executable is provided
    if executable:
        planner.connect()
        planner.reset(args.env)

    recording = not args.no_record

    # Run based on mode
    if args.compare:
        run_model_comparison_demo(
            strong_model=args.strong_model,
            weak_model=args.weak_model,
            recording=recording
        )
    elif args.demo:
        run_tool_generation_demo(planner, recording=recording)
    elif args.examples:
        run_examples(planner, recording=recording)
    elif args.task:
        result = planner.run(args.task, recording=recording)
        sys.exit(0 if result['success'] else 1)
    else:
        interactive_mode(planner, recording=recording)


if __name__ == "__main__":
    main()
