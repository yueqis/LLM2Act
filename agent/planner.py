"""
Planner Module for VirtualHome Agent

The Planner integrates tool generation and reuse:
1. Observe the scene to get available objects
2. Ask LLM to generate a plan using existing tools or define new ones
3. Validate and execute the plan
4. Save new tools to registry for future reuse
"""

import os
import re
import json
import time
import glob
import subprocess
from typing import List, Dict, Any, Optional, Tuple

# from openai import OpenAI
import openai

from tool_generator import Tool, SkillRegistry, ToolGenerator
from action_mapper import ACTIONS, UNITY_ACTIONS, format_action

try:
    from simulation.unity_simulator import comm_unity
except ImportError:
    print("Warning: VirtualHome API not found. Running in offline mode.")
    print("To enable simulation:")
    print("  git clone https://github.com/xavierpuigf/virtualhome.git")
    print("  export PYTHONPATH=$PYTHONPATH:/path/to/virtualhome/virtualhome")
    comm_unity = None

# Default configuration
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


class Planner:
    """
    VirtualHome Planner with integrated tool generation and reuse.

    The planner can:
    1. Use existing tools from the registry
    2. Generate new tools via LLM when needed
    3. Validate scripts before execution
    4. Execute scripts in VirtualHome simulator
    """

    def __init__(
        self,
        executable_path: str = "",
        port: str = "8080",
        model: str = None,
        init_basic_skills: bool = False,
    ):
        self.executable_path = executable_path
        self.port = port
        self.model = model or DEFAULT_MODEL
        self.comm = None
        self.character_id = 0
        self.task_counter = 0  # For unique video filenames

        # Initialize components
        self.registry = SkillRegistry(init_basic_skills=init_basic_skills)
        self.generator = ToolGenerator(model=self.model)

        # Setup LLM client
        api_key = DEFAULT_API_KEY
        base_url = DEFAULT_BASE_URL
        # if api_key and base_url:
        #     try: self.client = OpenAI(api_key=api_key, base_url=base_url)
        #     except: continue
        # else:
        #     self.client = None
        #     print("Warning: LLM not configured. Set OPENAI_API_KEY and OPENAI_BASE_URL.")

    # =========================================================================
    # Connection & Environment
    # =========================================================================

    def connect(self) -> bool:
        """Connect to VirtualHome simulator."""
        if not comm_unity:
            print("VirtualHome library not loaded.")
            return False

        print(f"Connecting to VirtualHome on port {self.port}...")
        # Get display from environment variable, default to "99" for headless
        x_display = os.getenv("DISPLAY", ":99").replace(":", "")
        self.comm = comm_unity.UnityCommunication(
            file_name=self.executable_path,
            port=self.port,
            x_display=x_display,
            timeout_wait=300  # 5 minutes timeout for video recording
        )
        print("Connected successfully.")
        return True

    def reset(self, env_id: int = 0):
        """Reset environment to a specific scene."""
        if self.comm:
            self.comm.reset(env_id)
            self.comm.add_character('Chars/Female1', initial_room='kitchen')
            time.sleep(1)
            print(f"Environment {env_id} reset.")

    def observe(self) -> Dict[str, Any]:
        """Get current scene observation."""
        if not self.comm:
            return {"objects": [], "agent_id": self.character_id}

        _, graph = self.comm.environment_graph()
        objects = []
        for node in graph['nodes']:
            objects.append({
                'name': node['class_name'],
                'id': node['id'],
                'states': node.get('states', [])
            })

        return {"objects": objects, "agent_id": self.character_id}

    # =========================================================================
    # Planning with Tool Reuse
    # =========================================================================

    def plan(self, task: str, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a plan for the task, reusing existing tools when possible.

        Returns:
            Dict with:
                - script: List of VirtualHome script lines
                - tools_used: List of tool names used
                - new_tools: List of newly generated tools
        """
        result = {"script": [], "tools_used": [], "new_tools": []}

        # if not self.client:
        #     print("LLM client not configured.")
        #     return result

        # Build object context
        objects = observation.get("objects", [])
        object_names = list(set(
            obj['name'] for obj in objects
            if obj['name'] not in ['floor', 'wall', 'ceiling', 'character']
        ))

        # Build object ID mapping
        object_map = {}  # name -> list of IDs
        for obj in objects:
            name = obj['name']
            if name not in object_map:
                object_map[name] = []
            object_map[name].append(obj['id'])

        # Build prompt
        prompt = self._build_plan_prompt(task, object_names, object_map)

        # Call LLM
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                # prompt=prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = response.choices[0].message.content
            result = self._parse_plan_response(content, object_map)
        except Exception as e:
            print(f"LLM error: {e}")

        return result

    def _build_plan_prompt(
        self,
        task: str,
        object_names: List[str],
        object_map: Dict[str, List[int]]
    ) -> str:
        """Build the planning prompt."""

        # Format available objects with IDs
        objects_str = "\n".join(
            f"  - {name}: IDs {ids}"
            for name, ids in sorted(object_map.items())[:100]
        )

        # Format available actions with descriptions
        actions_str = "\n".join(
            f"  - {action}: {ACTIONS[action]['params']} param(s), {ACTIONS[action]['name']}"
            for action in UNITY_ACTIONS
        )

        # Format available tools with full descriptions
        tools_list = self.registry.get_tools_for_prompt()
        has_tools = len(tools_list) > 0

        if has_tools:
            # Format tools with full details for better matching
            tools_json = json.dumps(tools_list, indent=2)
            tools_section = f"""AVAILABLE TOOLS (reusable action sequences):
{self.registry.get_tools_summary()}

TOOL DETAILS (for reference):
{tools_json}

INSTRUCTIONS:
1. First, check if any existing tool's DESCRIPTION matches what you need to do (or part of it)
2. If a tool's description matches your subtask, use "use_tool" with appropriate bindings
3. Tools are REUSABLE - the same tool can work with different objects via bindings
4. If no existing tool matches, create a new tool with "define_tool"
5. Break complex tasks into smaller reusable steps when possible

RULES FOR ACTIONS:
- WALK to an object before interacting with it
- GRAB an object before PUT/PUTIN operations
- OPEN a container before PUTIN"""
        else:
            tools_section = """AVAILABLE TOOLS: None (registry is empty)

INSTRUCTIONS:
1. Since no tools exist, you must create tools using "define_tool"
2. Design tools to be REUSABLE - use abstract parameter names like "object", "target", "container"
3. A good tool should be a useful building block that could help future tasks
4. Examples of good reusable tools:
   - "grab_and_place": Walk to object, grab it, walk to target, put it down
   - "put_in_container": Walk to object, grab it, walk to container, open, put in, close
   - "switch_appliance_on": Walk to appliance, switch it on

RULES FOR ACTIONS:
- WALK to an object before interacting with it
- GRAB an object before PUT/PUTIN operations
- OPEN a container before PUTIN"""

        return f"""You are a VirtualHome planner. Generate a plan to accomplish the task by creating or reusing tools.

TASK: {task}

AVAILABLE OBJECTS IN SCENE (name: IDs):
{objects_str}

AVAILABLE ACTIONS:
{actions_str}

{tools_section}

OUTPUT FORMAT (JSON only, no explanation):

When CREATING a new tool:
{{
  "plan": [
    {{
      "type": "define_tool",
      "name": "grab_and_place",
      "description": "Walk to an object, grab it, walk to target surface, and place object on it",
      "parameters": ["object", "target"],
      "actions": [
        {{"action": "WALK", "object": "object"}},
        {{"action": "GRAB", "object": "object"}},
        {{"action": "WALK", "object": "target"}},
        {{"action": "PUTBACK", "object": "object", "target": "target"}}
      ],
      "bindings": {{
        "object": "apple",
        "object_id": 123,
        "target": "table",
        "target_id": 456
      }}
    }}
  ]
}}

When REUSING an existing tool (use the EXACT parameter names from the tool definition):
{{
  "plan": [
    {{
      "type": "use_tool",
      "tool": "grab_and_place",
      "bindings": {{
        "object": "banana",
        "object_id": 316,
        "target": "coffeetable",
        "target_id": 113
      }}
    }}
  ]
}}

CRITICAL:
- When reusing a tool, use the EXACT parameter names from the tool's "parameters" list
- For "grab_and_place" with parameters ["object", "target"], use "object" and "target" in bindings
- "bindings" format: {{"param_name": "object_name", "param_name_id": ID}}
- Check tool DESCRIPTIONS to decide if a tool can be reused
- Output ONLY valid JSON, no markdown or explanation
"""

    def _parse_plan_response(
        self,
        content: str,
        object_map: Dict[str, List[int]]
    ) -> Dict[str, Any]:
        """Parse LLM response into executable plan."""
        result = {"script": [], "tools_used": [], "new_tools": []}

        try:
            # Strip markdown if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            data = json.loads(content)
            plan_steps = data.get("plan", [])

            # Track tools defined in this plan to avoid double execution
            tools_defined_this_plan = set()

            for step in plan_steps:
                step_type = step.get("type")

                if step_type == "use_tool":
                    # Use existing tool
                    tool_name = step.get("tool")
                    bindings = step.get("bindings", {})

                    # Skip if we just defined this tool (already executed)
                    if tool_name in tools_defined_this_plan:
                        continue

                    if self.registry.has_tool(tool_name):
                        tool = self.registry.get_tool(tool_name)
                        # Extract bindings with object names
                        extracted = self._extract_bindings(bindings)
                        script_lines = tool.to_script(extracted)
                        result["script"].extend(script_lines)
                        result["tools_used"].append(tool_name)
                    else:
                        print(f"Warning: Tool '{tool_name}' not found")

                elif step_type == "action":
                    # Direct action
                    action = step.get("action", "").upper()
                    obj = step.get("object")
                    obj_id = step.get("object_id")
                    target = step.get("target")
                    target_id = step.get("target_id")

                    objects = []
                    if obj and obj_id:
                        objects.append((obj, obj_id))
                    if target and target_id:
                        objects.append((target, target_id))

                    line = format_action(action, objects, self.character_id)
                    if line:
                        result["script"].append(line)

                elif step_type == "define_tool":
                    # Define and use new tool
                    tool_name = step.get("name", "unnamed")

                    # Skip if tool already exists (LLM re-defined it)
                    if self.registry.has_tool(tool_name):
                        # Just use the existing tool
                        tool = self.registry.get_tool(tool_name)
                        bindings = step.get("bindings", {})
                        extracted = self._extract_bindings(bindings)
                        script_lines = tool.to_script(extracted)
                        result["script"].extend(script_lines)
                        result["tools_used"].append(tool_name)
                        continue

                    tool = Tool(
                        name=tool_name,
                        description=step.get("description", ""),
                        parameters=step.get("parameters", []),
                        actions=step.get("actions", [])
                    )

                    # Validate
                    self.generator._validate(tool)

                    if tool.is_validated:
                        # Register for future use
                        self.registry.add_tool(tool)
                        result["new_tools"].append(tool.name)
                        tools_defined_this_plan.add(tool_name)

                        # Execute now
                        bindings = step.get("bindings", {})
                        extracted = self._extract_bindings(bindings)
                        # Debug: print bindings if script is empty
                        script_lines = tool.to_script(extracted)
                        if not script_lines or any('[Walk]' in l and '(' not in l for l in script_lines):
                            print(f"Debug: bindings={bindings}, extracted={extracted}")
                        result["script"].extend(script_lines)
                        result["tools_used"].append(tool_name)
                    else:
                        print(f"Warning: Generated tool failed validation: {tool.validation_errors}")

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Content: {content[:500]}")
        except Exception as e:
            print(f"Parse error: {e}")

        return result

    def _extract_bindings(self, bindings: Dict[str, Any]) -> Dict[str, Tuple[str, int]]:
        """
        Extract parameter -> (object_name, object_id) mappings from bindings.

        Expects bindings like: {"target": "tv", "target_id": 453}
        Returns: {"target": ("tv", 453)}
        """
        result = {}

        # Find all _id keys
        for key, value in bindings.items():
            if key.endswith("_id"):
                param_name = key[:-3]  # Remove "_id" suffix
                obj_name = bindings.get(param_name, param_name)  # Use param name as fallback
                try:
                    obj_id = int(value)
                    result[param_name] = (obj_name, obj_id)
                except (ValueError, TypeError):
                    pass

        return result

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_script(self, script: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate a VirtualHome script before execution.

        Returns:
            (is_valid, list of errors)
        """
        errors = []

        if not script:
            errors.append("Empty script")
            return False, errors

        # Parse and validate each line
        # Pattern allows action names with spaces like "Switch on"
        pattern = r'<char(\d+)> \[([^\]]+)\](?: <([^>]+)> \((\d+)\))?(?: <([^>]+)> \((\d+)\))?'

        for i, line in enumerate(script):
            match = re.match(pattern, line, re.IGNORECASE)
            if not match:
                errors.append(f"Line {i+1}: Invalid format '{line}'")
                continue

            action_name = match.group(2)
            # Normalize action name for lookup (e.g., "Switch on" -> "SWITCHON")
            action = action_name.upper().replace(" ", "")

            # Check action exists
            if action not in ACTIONS:
                errors.append(f"Line {i+1}: Unknown action '{action}'")
                continue

            # Check Unity support
            if action not in UNITY_ACTIONS:
                errors.append(f"Line {i+1}: Action '{action}' not supported in Unity")

        return len(errors) == 0, errors

    # =========================================================================
    # Execution
    # =========================================================================

    def execute(self, script: List[str], recording: bool = True) -> Dict[str, Any]:
        """Execute script in VirtualHome."""
        result = {'success': False, 'message': ''}

        if not self.comm:
            print("Not connected to simulator. Script:")
            for line in script:
                print(f"  {line}")
            result['message'] = 'Not connected'
            return result

        if not script:
            result['message'] = 'Empty script'
            return result

        # Validate first
        is_valid, errors = self.validate_script(script)
        if not is_valid:
            print("Script validation failed:")
            for err in errors:
                print(f"  - {err}")
            result['message'] = f'Validation failed: {errors}'
            return result

        print(f"Executing {len(script)} actions...")

        # Use unique prefix for each task
        task_prefix = f"task_{self.task_counter:03d}"
        self.task_counter += 1

        if recording:
            success, message = self.comm.render_script(
                script,
                recording=True,
                skip_animation=False,
                output_folder='Output/',
                file_name_prefix=task_prefix,
                image_synthesis=['normal'],
                frame_rate=25,
                camera_mode=['PERSON_FROM_BACK']
            )
            if success:
                vh_parent = os.path.dirname(self.executable_path)
                frames_dir = os.path.join(vh_parent, 'Output', task_prefix, '0')
                video_path = self._frames_to_video(frames_dir, f'{task_prefix}.mp4')
                if video_path:
                    print(f"Video saved to: {video_path}")
        else:
            success, message = self.comm.render_script(
                script, recording=False, skip_animation=False
            )

        result['success'] = success
        result['message'] = message

        if success:
            print("Execution successful.")
        else:
            print(f"Execution failed: {message}")

        return result

    def _frames_to_video(self, frames_dir: str, output_name: str, fps: int = 25) -> Optional[str]:
        """Convert PNG frames to MP4 video."""
        pattern = os.path.join(frames_dir, 'Action_*_normal.png')
        frames = sorted(glob.glob(pattern))

        if not frames:
            print(f"No frames found in {frames_dir}")
            return None

        # Get the starting frame number from the first file
        # Filename format: Action_XXXX_0_normal.png
        first_frame = os.path.basename(frames[0])
        start_number = int(first_frame.split('_')[1])

        output_path = os.path.join(frames_dir, output_name)

        try:
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-start_number', str(start_number),
                '-i', os.path.join(frames_dir, 'Action_%04d_0_normal.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg error: {e.stderr.decode() if e.stderr else e}")
            return None
        except FileNotFoundError:
            print("ffmpeg not found. Install with: brew install ffmpeg")
            return None

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def run(self, task: str, recording: bool = True) -> Dict[str, Any]:
        """
        Main entry: Observe -> Plan -> Validate -> Execute.

        Args:
            task: Natural language task description
            recording: Whether to record video

        Returns:
            Dict with 'success', 'message', 'script', 'tools_used', 'new_tools'
        """
        # Connect if needed
        if not self.comm and self.executable_path:
            self.connect()

        observation = self.observe()
        return self.run_with_observation(task, observation, recording)

    def run_with_observation(
        self, task: str, observation: Dict[str, Any], recording: bool = True
    ) -> Dict[str, Any]:
        """
        Run with a provided observation (useful for offline testing).

        Args:
            task: Natural language task description
            observation: Scene observation dict with 'objects' list
            recording: Whether to record video

        Returns:
            Dict with 'success', 'message', 'script', 'tools_used', 'new_tools'
        """
        result = {
            'success': False,
            'message': '',
            'script': [],
            'tools_used': [],
            'new_tools': []
        }

        # 1. Observe
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print('='*60)

        print(f"Found {len(observation.get('objects', []))} objects in scene")

        # 2. Plan
        print("\nPlanning...")
        plan_result = self.plan(task, observation)

        result['script'] = plan_result['script']
        result['tools_used'] = plan_result['tools_used']
        result['new_tools'] = plan_result['new_tools']

        print(f"\nGenerated script ({len(result['script'])} actions):")
        for line in result['script']:
            print(f"  {line}")

        if result['tools_used']:
            print(f"\nTools used: {result['tools_used']}")
        if result['new_tools']:
            print(f"New tools created: {result['new_tools']}")

        # 3. Execute
        if result['script']:
            print("\nExecuting...")
            exec_result = self.execute(result['script'], recording=recording)
            result['success'] = exec_result['success']
            result['message'] = exec_result['message']
        else:
            result['message'] = 'No actions generated'

        return result


# =============================================================================
# Tests
# =============================================================================

def test_planner_offline():
    """Test planner in offline mode (no simulator)."""
    planner = Planner(init_basic_skills=False)

    # Test with mock observation
    observation = {
        "objects": [
            {"name": "apple", "id": 101, "states": []},
            {"name": "table", "id": 202, "states": []},
            {"name": "fridge", "id": 303, "states": ["CLOSED"]},
        ],
        "agent_id": 0
    }

    # Test planning (requires LLM)
    if planner.client:
        result = planner.plan("grab the apple and put it on the table", observation)
        print(f"Plan result: {result}")
        print(f"Tools generated: {planner.registry.list_tools()}")
    else:
        print("LLM not configured, skipping plan test")

    # Test script validation
    valid_script = [
        "<char0> [Walk] <apple> (101)",
        "<char0> [Grab] <apple> (101)",
    ]
    is_valid, errors = planner.validate_script(valid_script)
    assert is_valid, f"Valid script should pass: {errors}"

    invalid_script = [
        "<char0> [FlyAway] <apple> (101)",
    ]
    is_valid, errors = planner.validate_script(invalid_script)
    assert not is_valid, "Invalid script should fail"

    print("test_planner_offline: PASSED")


def test_empty_registry():
    """Test that registry starts empty by default."""
    planner = Planner()

    # Registry should be empty by default
    assert len(planner.registry.list_tools()) == 0, "Registry should start empty"
    print("test_empty_registry: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Planner Tests")
    print("=" * 60)

    test_empty_registry()
    test_planner_offline()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
