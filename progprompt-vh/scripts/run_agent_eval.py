"""
Evaluation script for VirtualHome Agent on ProgPrompt-VH benchmark.

This script evaluates the agent implementation (from agent/) on the VirtualHome
tasks using the ProgPrompt-VH evaluation framework.

It reuses the evaluation infrastructure from run_eval.py but replaces
the LLM-based plan generation with the agent's planning system.

Usage:
    python scripts/run_agent_eval.py \\
        --progprompt-path $(pwd) \\
        --expt-name agent_eval \\
        --unity-filename ../macos_exec.v2.3.0.app \\
        --test-set test_seen \\
        --agent-model gpt-4o \\
        --init-basic-skills
"""

import sys
sys.path.append("virtualhome/simulation")
sys.path.append("virtualhome/demo")
sys.path.append("virtualhome")

import argparse
import os
import os.path as osp
import json
import time

from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from agent_adapter import AgentAdapter
from utils_execute import run_execution, current_state_prompt

# Import eval function from run_eval
from run_eval import eval


def agent_planner_executor(args):
    """
    Main evaluation function using the agent.
    
    This function:
    1. Initializes VirtualHome environment
    2. Loads test tasks
    3. Uses AgentAdapter to generate plans
    4. Executes plans in VirtualHome
    5. Evaluates results against ground truth
    """
    
    # Initialize environment
    print(f"\n{'='*70}")
    print("Initializing VirtualHome Environment")
    print('='*70)
    
    comm = UnityCommunication(
        file_name=args.unity_filename,
        port=args.port,
        x_display=args.display
    )
    
    # Set to environment 0 for initial setup
    comm.reset(0)
    _, env_graph = comm.environment_graph()
    obj = list(set([node['class_name'] for node in env_graph["nodes"]]))
    
    print(f"Environment initialized with {len(obj)} object types")
    
    # Initialize agent adapter
    print(f"\n{'='*70}")
    print("Initializing Agent")
    print('='*70)
    
    agent = AgentAdapter(
        comm=comm,
        model=args.agent_model,
        port=args.port,
        init_basic_skills=args.init_basic_skills
    )
    
    # Switch to test environment if needed
    if args.env_id != 0:
        comm.reset(args.env_id)
        _, graph = comm.environment_graph()
        obj = list(set([node['class_name'] for node in graph["nodes"]]))
        print(f"Switched to environment {args.env_id} with {len(obj)} object types")
        
        # Load test tasks for new environment
        test_tasks = []
        with open(f"{args.progprompt_path}/data/new_env/{args.test_set}_annotated.json", 'r') as f:
            for line in f.readlines():
                test_tasks.append(list(json.loads(line).keys())[0])
    else:
        # Load test tasks for environment 0
        test_tasks = []
        for file in os.listdir(f"{args.progprompt_path}/data/{args.test_set}"):
            with open(f"{args.progprompt_path}/data/{args.test_set}/{file}", 'r') as f:
                for line in f.readlines():
                    test_tasks.append(list(json.loads(line).keys())[0])
    
    print(f"\n{'='*70}")
    print(f"Loaded {len(test_tasks)} test tasks from {args.test_set}")
    print('='*70)
    
    # Setup logging
    log_filename = f"{args.expt_name}_agent_{args.agent_model.replace('/', '_')}_{args.test_set}"
    if args.init_basic_skills:
        log_filename += "_with_basic_skills"
    
    log_file = open(f"{args.progprompt_path}/results/{log_filename}_logs.txt", 'w')
    log_file.write(f"{'='*70}\n")
    log_file.write(f"Agent Evaluation on VirtualHome\n")
    log_file.write(f"{'='*70}\n")
    log_file.write(f"Model: {args.agent_model}\n")
    log_file.write(f"Test set: {args.test_set}\n")
    log_file.write(f"Environment ID: {args.env_id}\n")
    log_file.write(f"Basic skills initialized: {args.init_basic_skills}\n")
    log_file.write(f"Total tasks: {len(test_tasks)}\n")
    log_file.write(f"{'='*70}\n")
    log_file.write(f"\nTest tasks:\n")
    for i, task in enumerate(test_tasks):
        log_file.write(f"{i+1}. {task}\n")
    
    # Generate plans using agent
    if not args.load_generated_plans:
        print(f"\n{'='*70}")
        print("Generating Plans with Agent")
        print('='*70)
        
        gen_plan = []
        all_metadata = []
        
        for i, task in enumerate(test_tasks):
            print(f"\n[{i+1}/{len(test_tasks)}] Generating plan for: {task}")
            
            # Reset environment for each task
            agent.reset_environment(args.env_id)
            time.sleep(1)
            
            # Generate plan using agent
            plan_text, metadata = agent.generate_plan(task, log_file)
            gen_plan.append(plan_text)
            all_metadata.append(metadata)
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.5)
        
        # Save generated plans
        print(f"\nSaving generated plans to: {log_filename}_plans.json")
        with open(f"{args.progprompt_path}/results/{log_filename}_plans.json", 'w') as f:
            plans_dict = {}
            for task, plan, meta in zip(test_tasks, gen_plan, all_metadata):
                plans_dict[task] = {
                    'plan': plan,
                    'tools_used': meta['tools_used'],
                    'new_tools': meta['new_tools']
                }
            json.dump(plans_dict, f, indent=2)
        
        # Save agent's tool registry
        registry_file = f"{args.progprompt_path}/results/{log_filename}_tool_registry.json"
        agent.save_tool_registry(registry_file)
        
        log_file.write(f"\n{'='*70}\n")
        log_file.write(f"Agent Tool Registry Summary:\n")
        log_file.write(agent.get_tool_summary())
        log_file.write(f"\n{'='*70}\n")
    
    # Load from file if specified
    else:
        print(f"Loading generated plans from: {log_filename}_plans.json")
        with open(f"{args.progprompt_path}/results/{log_filename}_plans.json", 'r') as f:
            data = json.load(f)
            test_tasks, gen_plan = [], []
            for k, v in data.items():
                test_tasks.append(k)
                # Handle both old format (string) and new format (dict)
                if isinstance(v, dict):
                    gen_plan.append(v.get('plan', ''))
                else:
                    gen_plan.append(v)
    
    # Run execution
    print(f"\n{'='*70}")
    print("Running Execution in VirtualHome")
    print('='*70)
    
    log_file.write(f"\n----PROMPT for state check----\n{current_state_prompt}\n")
    
    final_states, initial_states, exec_per_task = run_execution(
        args,
        comm,
        test_tasks,
        gen_plan,
        log_file
    )
    
    # Evaluate
    print(f"\n{'='*70}")
    print("Evaluating Results")
    print('='*70)
    
    final_states_GT = []
    with open(f'{args.progprompt_path}/data/final_states/final_states_{args.test_set}.json', 'r') as f:
        for line in f.readlines():
            final_states_GT.append(json.loads(line))
    
    results = eval(
        final_states,
        final_states_GT,
        initial_states,
        test_tasks,
        exec_per_task,
        log_file
    )
    
    # Print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print('='*70)
    print(f"Model: {args.agent_model}")
    print(f"Test set: {args.test_set}")
    print(f"Basic skills: {args.init_basic_skills}")
    print()
    print(f"PSR (Partial Success Rate): {results['overall']['PSR']:.3f}")
    print(f"SR (Success Rate): {results['overall']['SR']:.3f}")
    print(f"Precision: {results['overall']['Precision']:.3f}")
    print(f"Exec (Executability): {results['overall']['Exec']:.3f}")
    print('='*70)
    
    # Save results
    with open(f"{args.progprompt_path}/results/{log_filename}_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    log_file.write(f"\n{'='*70}\n")
    log_file.write(f"FINAL RESULTS\n")
    log_file.write(f"{'='*70}\n")
    log_file.write(f"PSR: {results['overall']['PSR']:.3f}\n")
    log_file.write(f"SR: {results['overall']['SR']:.3f}\n")
    log_file.write(f"Precision: {results['overall']['Precision']:.3f}\n")
    log_file.write(f"Exec: {results['overall']['Exec']:.3f}\n")
    log_file.close()
    
    print(f"\nResults saved to: {log_filename}_metrics.json")
    print(f"Logs saved to: {log_filename}_logs.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VirtualHome Agent on ProgPrompt-VH benchmark"
    )
    
    # Required arguments
    parser.add_argument("--progprompt-path", type=str, required=True,
                        help="Path to progprompt-vh directory")
    parser.add_argument("--expt-name", type=str, required=True,
                        help="Experiment name for logging")
    
    # VirtualHome environment arguments
    parser.add_argument("--unity-filename", type=str,
                        default="/path/to/macos_exec.v2.3.0.app",
                        help="Path to VirtualHome Unity executable")
    parser.add_argument("--port", type=str, default="8000",
                        help="Unity communication port")
    parser.add_argument("--display", type=str, default="0",
                        help="X display number")
    parser.add_argument("--env-id", type=int, default=0,
                        help="Environment ID (0 for default)")
    
    # Agent arguments
    parser.add_argument("--agent-model", type=str, default="gpt-4o",
                        help="OpenAI model to use for agent")
    parser.add_argument("--init-basic-skills", action="store_true",
                        help="Initialize agent with basic skills")
    
    # Test set arguments
    parser.add_argument("--test-set", type=str, default="test_seen",
                        choices=['test_unseen', 'test_seen', 'test_unseen_ambiguous', 'env1', 'env2'],
                        help="Test set to evaluate on")
    
    # Optional arguments
    parser.add_argument("--load-generated-plans", action="store_true",
                        help="Load previously generated plans instead of generating new ones")
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    if not osp.isdir(f"{args.progprompt_path}/results/"):
        os.makedirs(f"{args.progprompt_path}/results/")
    
    agent_planner_executor(args=args)
