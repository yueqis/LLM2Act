# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""
This script evaluates plan generation using the VirtualHome Agent
for the VirtualHome environment tasks.

This replaces the LLM-based plan generation in run_eval.py with
the agent's tool-based planning system from agent/planner.py.

All other components (execution, evaluation) are reused from progprompt-vh.
"""

import sys
import os

# Add paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VH_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '../../virtualhome'))
AGENT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '../../agent'))

sys.path.insert(0, VH_PATH)
sys.path.insert(0, os.path.join(VH_PATH, 'virtualhome'))
sys.path.insert(0, AGENT_PATH)

import argparse
import os.path as osp
import random
import json
import time
import re
from typing import List

from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from virtualhome.demo.utils_demo import *

# Import from agent - reuse instead of reimplementing
from planner import Planner

from utils_execute import *
from run_eval import eval  # Reuse evaluation function from progprompt


def convert_to_pythonic_format(script: List[str]) -> str:
    """
    Convert VirtualHome action format to pythonic format.
    
    Agent outputs: "<char0> [Walk] <bedroom> (123)"
    ProgPrompt expects: "walk('bedroom')"
    
    Args:
        script: List of VirtualHome action strings
        
    Returns:
        Pythonic format script string
    """
    pythonic_lines = []
    
    for line in script:
        # Parse VirtualHome format: <char0> [Action] <obj1> (id1) <obj2> (id2)
        match = re.match(r'<char\d+>\s+\[(\w+)\](.+)?', line)
        if not match:
            continue
            
        action = match.group(1).lower()
        rest = match.group(2).strip() if match.group(2) else ""
        
        # Extract object names (ignore IDs)
        objects = re.findall(r'<(\w+)>\s+\(\d+\)', rest)
        
        # Format as pythonic
        if len(objects) == 0:
            pythonic_lines.append(f"\t{action}()")
        elif len(objects) == 1:
            pythonic_lines.append(f"\t{action}('{objects[0]}')")
        elif len(objects) == 2:
            pythonic_lines.append(f"\t{action}('{objects[0]}', '{objects[1]}')")
        else:
            # Fallback for more objects
            obj_str = ', '.join(f"'{obj}'" for obj in objects)
            pythonic_lines.append(f"\t{action}({obj_str})")
    
    return '\n'.join(pythonic_lines)


def planner_executer(args):
    """
    Main planner-executor function using agent instead of LLM.
    
    This function replaces the LLM-based plan generation in run_eval.py
    with the agent's tool-based planning system.
    """
    
    # Initialize env - same as run_eval.py
    comm = UnityCommunication(file_name=args.unity_filename, 
                              port=args.port, 
                              x_display=args.display)
    
    # Prompt example environment is set to env_id 0
    comm.reset(0)

    _, env_graph = comm.environment_graph()
    obj = list(set([node['class_name'] for node in env_graph["nodes"]]))

    print(f"Environment initialized with {len(obj)} object types")

    # Initialize agent's Planner instead of using LLM prompts
    # Reuse the Planner class from agent/planner.py
    print(f"\nInitializing agent planner...")
    print(f"Model: {args.agent_model}")
    print(f"Init basic skills: {args.init_basic_skills}")
    
    planner = Planner(
        executable_path="",  # We use progprompt's comm, not agent's own
        port=args.port,
        model=args.agent_model,
        init_basic_skills=args.init_basic_skills
    )
    
    # Set the communication instance from progprompt-vh
    planner.comm = comm
    planner.character_id = 0
    
    print(f"Agent initialized with {len(planner.registry.list_tools())} tools")

    # Evaluate in given unseen env
    if args.env_id != 0:
        comm.reset(args.env_id)
        _, graph = comm.environment_graph()
        obj = list(set([node['class_name'] for node in graph["nodes"]]))

        # Evaluation tasks in given unseen env
        test_tasks = []
        with open(f"{args.progprompt_path}/data/new_env/{args.test_set}_annotated.json", 'r') as f:
            for line in f.readlines():
                test_tasks.append(list(json.loads(line).keys())[0])

    # Setup logging
    log_filename = f"{args.expt_name}_agent_{args.agent_model.replace('/', '_')}_{args.test_set}"
    if args.init_basic_skills:
        log_filename += "_with_basic_skills"
    
    log_file = open(f"{args.progprompt_path}/results/{log_filename}_logs.txt", 'w')
    
    if args.env_id != 0:
        log_file.write(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")
    
    # Evaluate in seen env
    if args.env_id == 0:
        test_tasks = []
        for file in os.listdir(f"{args.progprompt_path}/data/{args.test_set}"):
            with open(f"{args.progprompt_path}/data/{args.test_set}/{file}", 'r') as f:
                for line in f.readlines():
                    test_tasks.append(list(json.loads(line).keys())[0])

        log_file.write(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")
    
    # test_tasks = test_tasks[:2]  ## Uncomment to limit for sample evaluation

    # Generate plans using agent instead of LLM
    # =========================================================================
    # KEY INTEGRATION POINT: This replaces lines 188-210 in run_eval.py
    # OLD: LM(curr_prompt, args.gpt_version, max_tokens=600, ...)
    # NEW: planner.plan(task, observation)
    # =========================================================================
    if not args.load_generated_plans:
        gen_plan = []
        all_metadata = []
        
        for i, task in enumerate(test_tasks):
            print(f"\n[{i+1}/{len(test_tasks)}] Generating plan for: {task}")
            
            # Reset environment for each task
            comm.reset(args.env_id)
            time.sleep(0.5)
            
            # Get observation from environment
            observation = planner.observe()
            
            log_file.write(f"\n\n{'='*60}\n")
            log_file.write(f"Task: {task}\n")
            log_file.write(f"Objects in scene: {len(observation.get('objects', []))}\n")
            
            # Use agent to generate plan - this is the key replacement
            # Instead of calling LM() function, we call planner.plan()
            plan_result = planner.plan(task, observation)
            
            # Extract plan information
            script = plan_result.get('script', [])
            tools_used = plan_result.get('tools_used', [])
            new_tools = plan_result.get('new_tools', [])
            
            print(f"Generated {len(script)} actions")
            if tools_used:
                print(f"Tools used: {tools_used}")
            if new_tools:
                print(f"New tools created: {new_tools}")
            
            log_file.write(f"Generated script ({len(script)} actions):\n")
            for line in script:
                log_file.write(f"  {line}\n")
            if tools_used:
                log_file.write(f"Tools used: {tools_used}\n")
            if new_tools:
                log_file.write(f"New tools created: {new_tools}\n")
            
            # Convert to pythonic format expected by progprompt execution
            plan_text = convert_to_pythonic_format(script)
            gen_plan.append(plan_text)
            
            # Store metadata for later inspection
            all_metadata.append({
                'script': script,
                'tools_used': tools_used,
                'new_tools': new_tools
            })
            
            log_file.write(f"\nConverted to pythonic format:\n{plan_text}\n")
            
            # Small delay between tasks
            time.sleep(0.5)

        # Save generated plans
        print(f"\nSaving generated plan at: {log_filename}_plans.json\n")
        with open(f"{args.progprompt_path}/results/{log_filename}_plans.json", 'w') as f:
            plans_dict = {}
            for task, plan, meta in zip(test_tasks, gen_plan, all_metadata):
                plans_dict[task] = {
                    'plan': plan,
                    'tools_used': meta['tools_used'],
                    'new_tools': meta['new_tools'],
                    'script': meta['script']
                }
            json.dump(plans_dict, f, indent=2)
        
        # Save agent's tool registry for inspection
        registry_file = f"{args.progprompt_path}/results/{log_filename}_tool_registry.json"
        planner.registry.save(registry_file)
        print(f"Saved tool registry to: {registry_file}\n")

    # Load from file
    else:
        print(f"Loading generated plan from: {log_filename}_plans.json\n")
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

    
    log_file.write(f"\n----PROMPT for state check----\n{current_state_prompt}\n")

    # Run execution - reuse from progprompt-vh
    print(f"\n----Running execution----\n")
    final_states, initial_states, exec_per_task = run_execution(args, 
                                                                comm, 
                                                                test_tasks, 
                                                                gen_plan,
                                                                log_file)
    

    # Evaluate - reuse from progprompt-vh
    final_states_GT = []
    with open(f'{args.progprompt_path}/data/final_states/final_states_{args.test_set}.json', 'r') as f:
        for line in f.readlines():
            final_states_GT.append((json.loads(line)))

    results = eval(final_states, 
         final_states_GT, 
         initial_states, 
         test_tasks,
         exec_per_task,
         log_file)

    print(f"\n----Results----\n{results['overall']}\n")
    with open(f"{args.progprompt_path}/results/{log_filename}_metric.json", 'w') as f:
        json.dump(results, f, indent=2)
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VirtualHome Agent on ProgPrompt-VH benchmark"
    )
    
    # Arguments from run_eval.py (progprompt-vh)
    parser.add_argument("--progprompt-path", type=str, required=True,
                        help="Path to progprompt-vh directory")
    parser.add_argument("--expt-name", type=str, required=True,
                        help="Experiment name for logging")
    parser.add_argument("--unity-filename", type=str, 
                        default="/path/to/macos_exec.v2.3.0.app",
                        help="Path to VirtualHome Unity executable")
    parser.add_argument("--port", type=str, default="8000",
                        help="Unity communication port")
    parser.add_argument("--display", type=str, default="0",
                        help="X display number for Unity")
    parser.add_argument("--env-id", type=int, default=0,
                        help="Environment ID (0 for default)")
    parser.add_argument("--test-set", type=str, default="test_unseen", 
                        choices=['test_unseen', 'test_seen', 'test_unseen_ambiguous', 'env1', 'env2'],
                        help="Test set to evaluate")
    parser.add_argument("--load-generated-plans", action="store_true",
                        help="Load previously generated plans")
    parser.add_argument("--prompt-task-examples-ablation", type=str, default="none",
                        choices=['none', 'no_comments', "no_feedback", "no_comments_feedback"],
                        help="Ablation setting (needed for execution compatibility)")
    
    # Arguments from agent.py - preserving agent configuration
    parser.add_argument("--agent-model", type=str, 
                        default=os.getenv("OPENAI_MODEL", "gpt-4o"),
                        help="OpenAI model to use for agent (default: gpt-4o)")
    parser.add_argument("--init-basic-skills", action="store_true",
                        help="Initialize agent with basic skills")
    parser.add_argument("--no-record", action="store_true",
                        help="Disable video recording (from agent.py)")
    
    # Additional agent.py features that could be supported in future:
    # --examples, --demo, --compare, --strong-model, --weak-model, --offline
    # These are not currently used in evaluation but preserved for compatibility
    
    args = parser.parse_args()

    # Create results directory if needed
    if not osp.isdir(f"{args.progprompt_path}/results/"):
        os.makedirs(f"{args.progprompt_path}/results/")

    planner_executer(args=args)
