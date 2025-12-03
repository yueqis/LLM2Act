"""
Adapter for integrating the VirtualHome Agent with ProgPrompt-VH evaluation framework.

This adapter bridges the gap between:
1. The agent implementation (agent/planner.py) which uses modern OpenAI API
2. The progprompt-vh evaluation framework which expects specific plan formats

The adapter:
- Initializes the agent's Planner with progprompt-vh's Unity connection
- Calls the agent to generate plans for tasks
- Converts agent output to progprompt-vh's expected script format
- Handles the differences in API versions and data structures
"""

import sys
import os

# Add agent directory to path
AGENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../agent'))
sys.path.insert(0, AGENT_PATH)

from typing import Dict, Any, List, Tuple
import re


class AgentAdapter:
    """
    Adapter class to interface between agent implementation and progprompt-vh.
    """
    
    def __init__(self, comm, model: str = "gpt-4o", port: str = "8000", init_basic_skills: bool = True):
        """
        Initialize the agent adapter.
        
        Args:
            comm: VirtualHome UnityCommunication instance from progprompt-vh
            model: OpenAI model to use
            port: Unity communication port
            init_basic_skills: Whether to initialize with basic skills
        """
        # Import here to avoid issues if virtualhome not in path
        from planner import Planner
        
        self.comm = comm
        self.model = model
        self.port = port
        
        # Initialize the agent's Planner
        # We pass empty executable_path since we'll use progprompt-vh's comm directly
        self.planner = Planner(
            executable_path="",
            port=port,
            model=model,
            init_basic_skills=init_basic_skills
        )
        
        # Set the communication instance from progprompt-vh
        self.planner.comm = comm
        self.planner.character_id = 0  # Will be set after add_character
        
        print(f"AgentAdapter initialized with model: {model}")
        print(f"Agent has {len(self.planner.registry.list_tools())} tools in registry")
    
    def reset_environment(self, env_id: int = 0):
        """Reset the environment and add character."""
        self.comm.reset(env_id)
        # Note: progprompt-vh adds character in run_execution, so we don't do it here
        # self.comm.add_character('Chars/Male2', initial_room='kitchen')
        
    def generate_plan(self, task: str, log_file=None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a plan for the given task using the agent.
        
        Args:
            task: Natural language task description
            log_file: Optional file handle for logging
            
        Returns:
            Tuple of (plan_text, metadata_dict)
            - plan_text: The generated plan in pythonic format for progprompt-vh
            - metadata_dict: Dictionary with 'tools_used', 'new_tools', 'script'
        """
        print(f"\n{'='*60}")
        print(f"Agent generating plan for: {task}")
        print('='*60)
        
        # Get observation from environment
        observation = self.planner.observe()
        
        if log_file:
            log_file.write(f"\n\n{'='*60}\n")
            log_file.write(f"Task: {task}\n")
            log_file.write(f"Objects in scene: {len(observation.get('objects', []))}\n")
        
        # Call agent to generate plan
        # Note: We don't record video here as progprompt-vh handles execution separately
        result = self.planner.run_with_observation(task, observation, recording=False)
        
        # Extract plan information
        script = result.get('script', [])
        tools_used = result.get('tools_used', [])
        new_tools = result.get('new_tools', [])
        
        print(f"Generated {len(script)} actions")
        if tools_used:
            print(f"Tools used: {tools_used}")
        if new_tools:
            print(f"New tools created: {new_tools}")
        
        if log_file:
            log_file.write(f"Generated script ({len(script)} actions):\n")
            for line in script:
                log_file.write(f"  {line}\n")
            if tools_used:
                log_file.write(f"Tools used: {tools_used}\n")
            if new_tools:
                log_file.write(f"New tools created: {new_tools}\n")
        
        # Convert script to pythonic format expected by progprompt-vh
        # Agent outputs: "<char0> [Walk] <bedroom> (123)"
        # ProgPrompt expects: "walk('bedroom')"
        plan_text = self.convert_to_pythonic_format(script)
        
        if log_file:
            log_file.write(f"\nConverted to pythonic format:\n{plan_text}\n")
        
        metadata = {
            'script': script,
            'tools_used': tools_used,
            'new_tools': new_tools,
            'success': result.get('success', False)
        }
        
        return plan_text, metadata
    
    def convert_to_pythonic_format(self, script: List[str]) -> str:
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
            # Extract action and objects
            match = re.match(r'<char\d+>\s+\[(\w+)\](.+)?', line)
            if not match:
                continue
                
            action = match.group(1).lower()
            rest = match.group(2).strip() if match.group(2) else ""
            
            # Extract object names (ignore IDs)
            objects = re.findall(r'<(\w+)>\s+\(\d+\)', rest)
            
            # Format as pythonic
            if len(objects) == 0:
                pythonic_lines.append(f"{action}()")
            elif len(objects) == 1:
                pythonic_lines.append(f"{action}('{objects[0]}')")
            elif len(objects) == 2:
                pythonic_lines.append(f"{action}('{objects[0]}', '{objects[1]}')")
            else:
                # Fallback for more objects
                obj_str = ', '.join(f"'{obj}'" for obj in objects)
                pythonic_lines.append(f"{action}({obj_str})")
        
        return '\n'.join(pythonic_lines)
    
    def convert_script_to_progprompt_format(self, script: List[str]) -> str:
        """
        Convert agent's action script to progprompt-vh expected format.
        
        The agent already outputs in VirtualHome format like:
        '[WALK] <object> (id)'
        
        This is the same format expected by progprompt-vh, so minimal conversion needed.
        
        Args:
            script: List of action strings
            
        Returns:
            Formatted script string
        """
        # Join actions with newlines - progprompt-vh parses them line by line
        return '\n'.join(script)
    
    def save_tool_registry(self, filepath: str = "agent_skills.json"):
        """Save the agent's tool registry for inspection."""
        self.planner.registry.save(filepath)
        print(f"Saved agent's tool registry to: {filepath}")
        
    def get_tool_summary(self) -> str:
        """Get a summary of tools in the agent's registry."""
        return self.planner.registry.get_tools_summary()
