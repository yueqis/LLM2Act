"""
Tool Generator Module for VirtualHome

Generates and manages reusable skills (tools) for the VirtualHome agent.
Tools are abstract action sequences that can be instantiated with concrete object IDs.

Example:
    generator = ToolGenerator(model="gpt-4o")
    tool = generator.generate(
        task="make coffee",
        available_objects=["coffeemaker", "mug", "coffee"]
    )
    # tool.actions = [WALK coffeemaker, SWITCHON coffeemaker, WALK mug, GRAB mug, ...]
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# from openai import OpenAI
import openai

from action_mapper import ACTIONS, UNITY_ACTIONS, format_action


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Tool:
    """
    A reusable skill composed of a sequence of VirtualHome actions.

    Attributes:
        name: Skill name (e.g., "make_coffee")
        description: What this skill does
        parameters: List of object types needed (e.g., ["coffeemaker", "mug"])
        actions: List of (action, object_param, target_param) tuples
    """
    name: str
    description: str
    parameters: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    # Metadata
    is_validated: bool = False
    validation_errors: List[str] = field(default_factory=list)
    usage_count: int = 0
    success_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "actions": self.actions,
            "is_validated": self.is_validated,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
        }

    def to_script(self, bindings: Dict[str, Any], char_id: int = 0) -> List[str]:
        """
        Convert tool to VirtualHome script with concrete object IDs.

        Args:
            bindings: Maps parameter names to (object_name, object_id) tuples or just object_id
                      e.g., {"coffeemaker": ("coffeemaker", 201)} or {"coffeemaker": 201}
            char_id: Character ID (default 0)

        Returns:
            List of VirtualHome script lines
        """
        script = []
        for action in self.actions:
            action_name = action["action"]
            obj = action.get("object")
            target = action.get("target")

            objects = []
            if obj and obj in bindings:
                binding = bindings[obj]
                if isinstance(binding, tuple):
                    objects.append(binding)  # (name, id)
                else:
                    objects.append((obj, binding))  # fallback: use param name
            if target and target in bindings:
                binding = bindings[target]
                if isinstance(binding, tuple):
                    objects.append(binding)
                else:
                    objects.append((target, binding))

            line = format_action(action_name, objects, char_id)
            if line:
                script.append(line)

        return script

    def get_summary(self) -> str:
        """Get a brief summary for LLM prompt."""
        params_str = ", ".join(self.parameters) if self.parameters else "none"
        return f"{self.name}({params_str}): {self.description}"


# =============================================================================
# Skill Registry
# =============================================================================

class SkillRegistry:
    """
    Central repository for reusable tools/skills.
    Tools are matched by description similarity, not just name.
    """
    def __init__(self, init_basic_skills: bool = False):
        """
        Initialize the registry.

        Args:
            init_basic_skills: If True, pre-populate with basic skills.
                              Default is False (empty registry).
        """
        self._tools: Dict[str, Tool] = {}
        if init_basic_skills:
            self._init_basic_skills()

    def _init_basic_skills(self):
        """Initialize with basic Level-1 skills."""
        basic_skills = [
            Tool(
                name="walk_to",
                description="Walk to an object",
                parameters=["target"],
                actions=[{"action": "WALK", "object": "target"}],
                is_validated=True
            ),
            Tool(
                name="grab_object",
                description="Walk to and grab an object",
                parameters=["object"],
                actions=[
                    {"action": "WALK", "object": "object"},
                    {"action": "GRAB", "object": "object"}
                ],
                is_validated=True
            ),
            Tool(
                name="put_on",
                description="Put a held object onto a surface",
                parameters=["object", "surface"],
                actions=[
                    {"action": "WALK", "object": "surface"},
                    {"action": "PUTBACK", "object": "object", "target": "surface"}
                ],
                is_validated=True
            ),
            Tool(
                name="put_in",
                description="Put a held object into a container",
                parameters=["object", "container"],
                actions=[
                    {"action": "WALK", "object": "container"},
                    {"action": "OPEN", "object": "container"},
                    {"action": "PUTIN", "object": "object", "target": "container"},
                    {"action": "CLOSE", "object": "container"}
                ],
                is_validated=True
            ),
            Tool(
                name="turn_on",
                description="Walk to and switch on an appliance",
                parameters=["appliance"],
                actions=[
                    {"action": "WALK", "object": "appliance"},
                    {"action": "SWITCHON", "object": "appliance"}
                ],
                is_validated=True
            ),
            Tool(
                name="turn_off",
                description="Walk to and switch off an appliance",
                parameters=["appliance"],
                actions=[
                    {"action": "WALK", "object": "appliance"},
                    {"action": "SWITCHOFF", "object": "appliance"}
                ],
                is_validated=True
            ),
            Tool(
                name="sit_on",
                description="Walk to and sit on a sittable object",
                parameters=["seat"],
                actions=[
                    {"action": "WALK", "object": "seat"},
                    {"action": "SIT", "object": "seat"}
                ],
                is_validated=True
            ),
            Tool(
                name="open_object",
                description="Walk to and open a container",
                parameters=["container"],
                actions=[
                    {"action": "WALK", "object": "container"},
                    {"action": "OPEN", "object": "container"}
                ],
                is_validated=True
            ),
            Tool(
                name="close_object",
                description="Walk to and close a container",
                parameters=["container"],
                actions=[
                    {"action": "WALK", "object": "container"},
                    {"action": "CLOSE", "object": "container"}
                ],
                is_validated=True
            ),
        ]

        for tool in basic_skills:
            self._tools[tool.name] = tool

    def add_tool(self, tool: Tool) -> bool:
        """Add or update a tool in the registry."""
        self._tools[tool.name] = tool
        return True

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_tools_summary(self) -> str:
        """Get summary of all tools for LLM prompt with detailed descriptions."""
        if not self._tools:
            return "  (no tools available)"

        summaries = []
        for tool in self._tools.values():
            params_str = ", ".join(tool.parameters) if tool.parameters else "none"
            # Include detailed description and action sequence info
            actions_preview = " -> ".join(
                a.get("action", "?") for a in tool.actions[:4]
            )
            if len(tool.actions) > 4:
                actions_preview += " -> ..."
            summaries.append(
                f"  - {tool.name}({params_str})\n"
                f"      Description: {tool.description}\n"
                f"      Actions: {actions_preview}"
            )
        return "\n".join(summaries)

    def get_tools_for_prompt(self) -> List[Dict[str, Any]]:
        """Get tool information in structured format for LLM prompt."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "actions": tool.actions
            }
            for tool in self._tools.values()
        ]

    def save(self, filepath: str) -> bool:
        """Save all tools to a JSON file."""
        try:
            data = {
                "version": "1.0",
                "saved_at": datetime.now().isoformat(),
                "tools": [tool.to_dict() for tool in self._tools.values()]
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(self._tools)} tools to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving tools: {e}")
            return False

    def load(self, filepath: str) -> bool:
        """Load tools from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            tools_data = data.get("tools", [])
            loaded_count = 0
            for td in tools_data:
                tool = Tool(
                    name=td["name"],
                    description=td.get("description", ""),
                    parameters=td.get("parameters", []),
                    actions=td.get("actions", []),
                    is_validated=td.get("is_validated", True),
                    usage_count=td.get("usage_count", 0),
                    success_count=td.get("success_count", 0)
                )
                self._tools[tool.name] = tool
                loaded_count += 1

            print(f"Loaded {loaded_count} tools from {filepath}")
            return True
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return False
        except Exception as e:
            print(f"Error loading tools: {e}")
            return False

    def clear(self):
        """Clear all tools from registry."""
        self._tools.clear()


# =============================================================================
# Tool Generator
# =============================================================================

class ToolGenerator:
    """
    Generates new tools using LLM based on task description and available objects.
    """

    def __init__(self, model: str = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "")

        # if api_key and base_url:
        #     self.client = OpenAI(api_key=api_key, base_url=base_url)
        # else:
        #     self.client = None

    def _get_actions_description(self) -> str:
        """Get formatted description of available Unity actions."""
        lines = []
        for action_key in UNITY_ACTIONS:
            info = ACTIONS[action_key]
            params = info['params']
            if params == 0:
                params_desc = "no object"
            elif params == 1:
                params_desc = "object"
            else:
                params_desc = "object, target"
            lines.append(f"  - {action_key}: {params_desc}")
        return "\n".join(lines)

    def generate(self, task: str, available_objects: List[str]) -> Tool:
        """
        Generate a new tool for the given task.

        Args:
            task: Task description (e.g., "make coffee")
            available_objects: List of object names in the scene

        Returns:
            Generated Tool (check is_validated for success)
        """
        tool_name = task.lower().replace(" ", "_")
        tool = Tool(name=tool_name, description=task)

        # if not self.client:
        #     tool.validation_errors.append("LLM client not configured")
        #     return tool

        prompt = f"""You are a VirtualHome action planner. Generate a tool (reusable skill) for this task.

TASK: {task}

AVAILABLE OBJECTS IN SCENE:
{', '.join(available_objects[:50])}

AVAILABLE ACTIONS:
{self._get_actions_description()}

RULES:
1. Use ONLY actions from the list above
2. Use object names from AVAILABLE OBJECTS as parameters
3. WALK to an object before interacting with it
4. GRAB an object before PUT operations
5. OPEN containers before PUTIN

OUTPUT FORMAT (JSON only, no explanation):
{{
  "name": "{tool_name}",
  "description": "Brief description",
  "parameters": ["object1", "object2"],
  "actions": [
    {{"action": "WALK", "object": "object1"}},
    {{"action": "GRAB", "object": "object1"}},
    {{"action": "WALK", "object": "object2"}},
    {{"action": "PUTBACK", "object": "object1", "target": "object2"}}
  ]
}}
"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                # prompt=prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = response.choices[0].message.content
            tool = self._parse_response(content, tool_name)
        except Exception as e:
            tool.validation_errors.append(f"LLM error: {str(e)}")

        # Validate the generated tool
        self._validate(tool)
        return tool

    def _parse_response(self, content: str, default_name: str) -> Tool:
        """Parse LLM response into a Tool object."""
        tool = Tool(name=default_name, description="")

        try:
            # Strip markdown code blocks if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            data = json.loads(content)

            tool.name = data.get("name", default_name)
            tool.description = data.get("description", "")
            tool.parameters = data.get("parameters", [])
            tool.actions = data.get("actions", [])

        except json.JSONDecodeError as e:
            tool.validation_errors.append(f"JSON parse error: {str(e)}")
        except Exception as e:
            tool.validation_errors.append(f"Parse error: {str(e)}")

        return tool

    def _validate(self, tool: Tool) -> bool:
        """Validate tool syntax and logic."""
        errors = []

        # Check basic structure
        if not tool.name:
            errors.append("Tool must have a name")
        if not tool.actions:
            errors.append("Tool must have at least one action")

        # Check each action
        walked_to = set()
        grabbed = set()
        opened = set()

        for i, action in enumerate(tool.actions):
            action_name = action.get("action", "").upper()
            obj = action.get("object")
            target = action.get("target")

            # Check action exists
            if action_name not in ACTIONS:
                errors.append(f"Action {i+1}: Unknown action '{action_name}'")
                continue

            # Check Unity support
            if action_name not in UNITY_ACTIONS:
                errors.append(f"Action {i+1}: '{action_name}' not supported in Unity")
                continue

            # Check parameters
            required_params = ACTIONS[action_name]['params']
            if required_params >= 1 and not obj:
                errors.append(f"Action {i+1}: '{action_name}' requires an object")
            if required_params == 2 and not target:
                errors.append(f"Action {i+1}: '{action_name}' requires a target")

            # Track state for logic validation
            if action_name in ['WALK', 'FIND']:
                if obj:
                    walked_to.add(obj)
            elif action_name == 'GRAB':
                if obj and obj not in walked_to:
                    errors.append(f"Action {i+1}: Must WALK to '{obj}' before GRAB")
                if obj:
                    grabbed.add(obj)
            elif action_name == 'OPEN':
                if obj and obj not in walked_to:
                    errors.append(f"Action {i+1}: Must WALK to '{obj}' before OPEN")
                if obj:
                    opened.add(obj)
            elif action_name in ['PUTBACK', 'PUTIN']:
                if obj and obj not in grabbed:
                    errors.append(f"Action {i+1}: Must GRAB '{obj}' before {action_name}")
                if action_name == 'PUTIN' and target and target not in opened:
                    errors.append(f"Action {i+1}: Must OPEN '{target}' before PUTIN")

        tool.validation_errors = errors
        tool.is_validated = len(errors) == 0
        return tool.is_validated


# =============================================================================
# Tests
# =============================================================================

def test_empty_registry():
    """Test that registry starts empty by default."""
    registry = SkillRegistry()

    # Registry should be empty by default
    assert len(registry.list_tools()) == 0, "Registry should start empty"
    assert registry.get_tools_summary() == "  (no tools available)"

    print("test_empty_registry: PASSED")


def test_registry_with_basic_skills():
    """Test registry when initialized with basic skills."""
    registry = SkillRegistry(init_basic_skills=True)

    # Check basic skills exist
    assert registry.has_tool("walk_to"), "Should have walk_to"
    assert registry.has_tool("grab_object"), "Should have grab_object"
    assert registry.has_tool("put_on"), "Should have put_on"

    # Test script generation
    tool = registry.get_tool("grab_object")
    script = tool.to_script({"object": 123})
    assert len(script) == 2, "grab_object should have 2 actions"
    assert "[Walk]" in script[0], "First action should be Walk"
    assert "[Grab]" in script[1], "Second action should be Grab"
    assert "(123)" in script[0], "Should have object ID"

    print("test_registry_with_basic_skills: PASSED")


def test_tool_validation():
    """Test tool validation."""
    generator = ToolGenerator()

    # Valid tool
    valid_tool = Tool(
        name="test",
        description="Test",
        parameters=["apple"],
        actions=[
            {"action": "WALK", "object": "apple"},
            {"action": "GRAB", "object": "apple"}
        ]
    )
    assert generator._validate(valid_tool), "Valid tool should pass"

    # Invalid: GRAB without WALK
    invalid_tool = Tool(
        name="test",
        description="Test",
        parameters=["apple"],
        actions=[
            {"action": "GRAB", "object": "apple"}
        ]
    )
    assert not generator._validate(invalid_tool), "Should fail without WALK"

    # Invalid: unknown action
    invalid_tool2 = Tool(
        name="test",
        description="Test",
        actions=[{"action": "FLY", "object": "apple"}]
    )
    assert not generator._validate(invalid_tool2), "Should fail with unknown action"

    print("test_tool_validation: PASSED")


def test_script_generation():
    """Test VirtualHome script generation."""
    tool = Tool(
        name="put_apple_on_table",
        description="Put apple on table",
        parameters=["apple", "table"],
        actions=[
            {"action": "WALK", "object": "apple"},
            {"action": "GRAB", "object": "apple"},
            {"action": "WALK", "object": "table"},
            {"action": "PUTBACK", "object": "apple", "target": "table"}
        ]
    )

    bindings = {"apple": 101, "table": 202}
    script = tool.to_script(bindings)

    assert len(script) == 4, "Should have 4 lines"
    assert "<apple> (101)" in script[0], "Should have apple ID"
    assert "<table> (202)" in script[2], "Should have table ID"
    assert "[Put]" in script[3], "Should have Put action"

    print("test_script_generation: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Tool Generator Tests")
    print("=" * 60)

    test_empty_registry()
    test_registry_with_basic_skills()
    test_tool_validation()
    test_script_generation()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)

    # Demo: Generate a new tool
    print("\n" + "=" * 60)
    print("Demo: Generating a new tool with LLM")
    print("=" * 60)

    generator = ToolGenerator()
    if generator.client:
        tool = generator.generate(
            task="watch TV",
            available_objects=["tv", "remote_control", "sofa", "table"]
        )
        print(f"\nGenerated Tool: {tool.name}")
        print(f"Description: {tool.description}")
        print(f"Parameters: {tool.parameters}")
        print(f"Actions: {json.dumps(tool.actions, indent=2)}")
        print(f"Validated: {tool.is_validated}")
        if tool.validation_errors:
            print(f"Errors: {tool.validation_errors}")

        if tool.is_validated:
            bindings = {"tv": 301, "remote_control": 302, "sofa": 303}
            script = tool.to_script(bindings)
            print(f"\nVirtualHome Script:")
            for line in script:
                print(f"  {line}")
    else:
        print("LLM client not configured. Set OPENAI_API_KEY and OPENAI_BASE_URL.")
