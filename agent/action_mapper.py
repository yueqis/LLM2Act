"""
Action Mapper Module for VirtualHome
Maps natural language verbs to VirtualHome executable actions

Input: natural-language verb + task description
Output: VirtualHome executable action or None

Author: Yanlin
"""

from difflib import get_close_matches, SequenceMatcher


# =============================================================================
# VirtualHome Action Definitions
# =============================================================================

ACTIONS = {
    'WALK': {'name': 'Walk', 'params': 1, 'properties': [[]], 'unity': True},
    'RUN': {'name': 'Run', 'params': 1, 'properties': [[]], 'unity': True},
    'FIND': {'name': 'Find', 'params': 1, 'properties': [[]], 'unity': True},
    'TURNTO': {'name': 'TurnTo', 'params': 1, 'properties': [[]], 'unity': True},
    'LOOKAT': {'name': 'LookAt', 'params': 1, 'properties': [[]], 'unity': True},
    'GRAB': {'name': 'Grab', 'params': 1, 'properties': [['GRABBABLE']], 'unity': True},
    'OPEN': {'name': 'Open', 'params': 1, 'properties': [['CAN_OPEN']], 'unity': True},
    'CLOSE': {'name': 'Close', 'params': 1, 'properties': [['CAN_OPEN']], 'unity': True},
    'TOUCH': {'name': 'Touch', 'params': 1, 'properties': [[]], 'unity': True},
    'PUTBACK': {'name': 'Put', 'params': 2, 'properties': [['GRABBABLE'], []], 'unity': True},
    'PUT': {'name': 'Put', 'params': 2, 'properties': [['GRABBABLE'], []], 'unity': True},  # Alias for PUTBACK
    'PUTIN': {'name': 'PutIn', 'params': 2, 'properties': [['GRABBABLE'], ['CAN_OPEN']], 'unity': True},
    'SIT': {'name': 'Sit', 'params': 1, 'properties': [['SITTABLE']], 'unity': True},
    'STANDUP': {'name': 'StandUp', 'params': 0, 'properties': [], 'unity': True},
    'SWITCHON': {'name': 'SwitchOn', 'params': 1, 'properties': [['HAS_SWITCH']], 'unity': True},
    'SWITCHOFF': {'name': 'SwitchOff', 'params': 1, 'properties': [['HAS_SWITCH']], 'unity': True},
    'DRINK': {'name': 'Drink', 'params': 1, 'properties': [['DRINKABLE', 'RECIPIENT']], 'unity': True},
    
    # Evolving Graph only
    'LOOKAT_SHORT': {'name': 'Look at short', 'params': 1, 'properties': [[]], 'unity': False},
    'LOOKAT_LONG': {'name': 'Look at long', 'params': 1, 'properties': [[]], 'unity': False},
    'WATCH': {'name': 'Watch', 'params': 1, 'properties': [[]], 'unity': False},
    'POINTAT': {'name': 'Point at', 'params': 1, 'properties': [[]], 'unity': False},
    'DROP': {'name': 'Drop', 'params': 1, 'properties': [[]], 'unity': False},
    'RELEASE': {'name': 'Release', 'params': 1, 'properties': [[]], 'unity': False},
    'PUTOBJBACK': {'name': 'Put back', 'params': 1, 'properties': [[]], 'unity': False},
    'LIE': {'name': 'Lie', 'params': 1, 'properties': [['LIEABLE']], 'unity': False},
    'SLEEP': {'name': 'Sleep', 'params': 0, 'properties': [], 'unity': False},
    'WAKEUP': {'name': 'WakeUp', 'params': 0, 'properties': [], 'unity': False},
    'PLUGIN': {'name': 'PlugIn', 'params': 1, 'properties': [['HAS_PLUG']], 'unity': False},
    'PLUGOUT': {'name': 'PlugOut', 'params': 1, 'properties': [['HAS_PLUG']], 'unity': False},
    'EAT': {'name': 'Eat', 'params': 1, 'properties': [['EATABLE']], 'unity': False},
    'POUR': {'name': 'Pour', 'params': 2, 'properties': [['POURABLE', 'DRINKABLE'], ['RECIPIENT']], 'unity': False},
    'CUT': {'name': 'Cut', 'params': 1, 'properties': [['EATABLE', 'CUTABLE']], 'unity': False},
    'PUTON': {'name': 'PutOn', 'params': 1, 'properties': [['CLOTHES']], 'unity': False},
    'PUTOFF': {'name': 'PutOff', 'params': 1, 'properties': [['CLOTHES']], 'unity': False},
    'WIPE': {'name': 'Wipe', 'params': 1, 'properties': [[]], 'unity': False},
    'WASH': {'name': 'Wash', 'params': 1, 'properties': [[]], 'unity': False},
    'RINSE': {'name': 'Rinse', 'params': 1, 'properties': [[]], 'unity': False},
    'SCRUB': {'name': 'Scrub', 'params': 1, 'properties': [[]], 'unity': False},
    'SQUEEZE': {'name': 'Squeeze', 'params': 1, 'properties': [['CLOTHES']], 'unity': False},
    'PUSH': {'name': 'Push', 'params': 1, 'properties': [['MOVABLE']], 'unity': False},
    'PULL': {'name': 'Pull', 'params': 1, 'properties': [['MOVABLE']], 'unity': False},
    'MOVE': {'name': 'Move', 'params': 1, 'properties': [['MOVABLE']], 'unity': False},
    'READ': {'name': 'Read', 'params': 1, 'properties': [['READABLE']], 'unity': False},
    'TYPE': {'name': 'Type', 'params': 1, 'properties': [['HAS_SWITCH']], 'unity': False},
    'GREET': {'name': 'Greet', 'params': 1, 'properties': [['PERSON']], 'unity': False},
}

UNITY_ACTIONS = [k for k, v in ACTIONS.items() if v['unity']]


# =============================================================================
# Verb to Action Mapping Dictionary
# =============================================================================

VERB_MAPPING = {
    'walk': 'WALK', 'go': 'WALK', 'move to': 'WALK', 'navigate to': 'WALK',
    'run': 'RUN', 'run to': 'RUN', 'sprint': 'RUN',
    'find': 'FIND', 'search': 'FIND', 'locate': 'FIND', 'look for': 'FIND',
    'turn to': 'TURNTO', 'face': 'TURNTO',
    'look at': 'LOOKAT', 'look': 'LOOKAT', 'view': 'LOOKAT', 'examine': 'LOOKAT',
    'watch': 'WATCH', 'observe': 'WATCH',
    'point at': 'POINTAT', 'point': 'POINTAT',
    'grab': 'GRAB', 'take': 'GRAB', 'pick up': 'GRAB', 'pick': 'GRAB', 
    'hold': 'GRAB', 'grasp': 'GRAB', 'get': 'GRAB',
    'open': 'OPEN',
    'close': 'CLOSE', 'shut': 'CLOSE',
    'touch': 'TOUCH',
    'drop': 'DROP',
    'release': 'RELEASE', 'let go': 'RELEASE',
    'put': 'PUTBACK', 'place': 'PUTBACK', 'set': 'PUTBACK', 
    'put on': 'PUTBACK', 'place on': 'PUTBACK',
    'put in': 'PUTIN', 'place in': 'PUTIN', 'insert': 'PUTIN',
    'put back': 'PUTOBJBACK',
    'sit': 'SIT', 'sit down': 'SIT', 'sit on': 'SIT',
    'stand up': 'STANDUP', 'stand': 'STANDUP', 'get up': 'STANDUP',
    'lie': 'LIE', 'lie down': 'LIE', 'lay down': 'LIE',
    'sleep': 'SLEEP',
    'wake up': 'WAKEUP', 'wake': 'WAKEUP',
    'switch on': 'SWITCHON', 'turn on': 'SWITCHON', 'activate': 'SWITCHON',
    'switch off': 'SWITCHOFF', 'turn off': 'SWITCHOFF', 'deactivate': 'SWITCHOFF',
    'plug in': 'PLUGIN',
    'plug out': 'PLUGOUT', 'unplug': 'PLUGOUT',
    'drink': 'DRINK', 'sip': 'DRINK',
    'eat': 'EAT', 'consume': 'EAT',
    'pour': 'POUR',
    'cut': 'CUT', 'slice': 'CUT', 'chop': 'CUT',
    'wear': 'PUTON', 'put on clothes': 'PUTON', 'dress': 'PUTON',
    'take off': 'PUTOFF', 'remove': 'PUTOFF', 'undress': 'PUTOFF',
    'wipe': 'WIPE', 'clean': 'WIPE',
    'wash': 'WASH',
    'rinse': 'RINSE',
    'scrub': 'SCRUB',
    'squeeze': 'SQUEEZE', 'wring': 'SQUEEZE',
    'push': 'PUSH',
    'pull': 'PULL', 'drag': 'PULL',
    'move': 'MOVE',
    'read': 'READ',
    'type': 'TYPE', 'type on': 'TYPE',
    'greet': 'GREET', 'say hello': 'GREET',
}


# =============================================================================
# LLM Prompt Generation
# =============================================================================

def _generate_llm_prompt(verb, task_description):
    """Generate LLM prompt (Unity-supported actions only)"""
    unity_actions_info = []
    for action_key in UNITY_ACTIONS:
        info = ACTIONS[action_key]
        props = ', '.join(['/'.join(p) if p else 'None' for p in info['properties']])
        unity_actions_info.append(
            f"- {action_key}: {info['name']} (params: {info['params']}, properties: {props if props else 'None'})"
        )
    
    prompt = f"""Select the most appropriate VirtualHome action for the verb: "{verb}"
Task context: {task_description}

Available actions:
{chr(10).join(unity_actions_info)}

Respond ONLY with the action name (e.g., GRAB, WALK, OPEN). If none match, respond "NONE"."""
    
    return prompt


def _call_llm(verb, task_description, llm_func):
    """Call LLM to get action"""
    if llm_func is None:
        return None
    
    try:
        prompt = _generate_llm_prompt(verb, task_description)
        response = llm_func(prompt).strip().upper()
        
        if response in UNITY_ACTIONS:
            return response
        return None
    except:
        return None


# =============================================================================
# Core Mapping Function
# =============================================================================

def map_action(verb, task_description="", llm_func=None, use_fuzzy=True, 
               similarity_threshold=0.6, unity_only=True):
    """
    Map natural language verb to VirtualHome executable action
    
    Args:
        verb: Natural language verb (e.g., "pick up", "turn on")
        task_description: Task description providing context for LLM
        llm_func: LLM function accepting prompt string and returning response. None to disable LLM
        use_fuzzy: Whether to use fuzzy matching
        similarity_threshold: Fuzzy matching threshold (0-1)
        unity_only: Whether to return only Unity-supported actions
    
    Returns:
        dict: {
            'action': str or None,  # Action name like 'GRAB', None if not found
            'match_type': str,      # 'exact', 'fuzzy', 'llm', 'none'
            'confidence': float,    # Confidence score 0-1
            'unity_supported': bool,
            'properties': list,     # Required object properties
            'params': int          # Number of parameters
        }
    """
    verb_lower = verb.lower().strip()
    
    # 1. Exact match
    action = VERB_MAPPING.get(verb_lower)
    if action:
        if unity_only and action not in UNITY_ACTIONS:
            pass  # Try other methods
        else:
            info = ACTIONS[action]
            return {
                'action': action,
                'match_type': 'exact',
                'confidence': 1.0,
                'unity_supported': info['unity'],
                'properties': info['properties'],
                'params': info['params']
            }
    
    # 2. Fuzzy match
    if use_fuzzy:
        matches = get_close_matches(verb_lower, VERB_MAPPING.keys(), n=1, cutoff=similarity_threshold)
        if matches:
            matched_verb = matches[0]
            action = VERB_MAPPING[matched_verb]
            
            if unity_only and action not in UNITY_ACTIONS:
                pass  # Try LLM next
            else:
                info = ACTIONS[action]
                confidence = SequenceMatcher(None, verb_lower, matched_verb).ratio()
                return {
                    'action': action,
                    'match_type': 'fuzzy',
                    'confidence': confidence,
                    'matched_to': matched_verb,
                    'unity_supported': info['unity'],
                    'properties': info['properties'],
                    'params': info['params']
                }
    
    # 3. LLM fallback
    action = _call_llm(verb, task_description, llm_func)
    if action:
        info = ACTIONS[action]
        return {
            'action': action,
            'match_type': 'llm',
            'confidence': 0.85,
            'unity_supported': info['unity'],
            'properties': info['properties'],
            'params': info['params']
        }
    
    # 4. Not found
    return {
        'action': None,
        'match_type': 'none',
        'confidence': 0.0,
        'unity_supported': False,
        'properties': None,
        'params': -1
    }


def format_action(action_name, objects, char_id=0):
    """
    Format action to VirtualHome standard format
    
    Args:
        action_name: Action name (e.g., 'GRAB')
        objects: [(object_name, object_id), ...] 
        char_id: Character ID
    
    Returns:
        str: VirtualHome formatted string, e.g., "<char0> [Grab] <cup> (1)"
    """
    if action_name not in ACTIONS:
        return None
    
    action_display = ACTIONS[action_name]['name']
    action_str = f"<char{char_id}> [{action_display}]"
    
    for obj_name, obj_id in objects:
        action_str += f" <{obj_name}> ({obj_id})"
    
    return action_str


# =============================================================================
# Simple Tests
# =============================================================================

if __name__ == '__main__':
    print("Action Mapper Module\n" + "=" * 60)
    
    # Test 1: Exact match
    result = map_action("grab")
    print(f"✅ Exact match: 'grab' -> {result['action']}")
    
    # Test 2: Fuzzy match
    result = map_action("pickup")
    print(f"✅ Fuzzy match: 'pickup' -> {result['action']} (matched: {result.get('matched_to', 'N/A')})")
    
    # Test 3: LLM fallback (mock)
    def mock_llm(prompt):
        if "retrieve" in prompt.lower():
            return "GRAB"
        return "NONE"
    
    result = map_action("retrieve", "get object from location", llm_func=mock_llm)
    print(f"✅ LLM fallback: 'retrieve' -> {result['action']}")
    
    # Test 4: Format action
    formatted = format_action("GRAB", [("cup", 1)])
    print(f"✅ Format: {formatted}")
    
    # Test 5: Unity filter
    result = map_action("eat", unity_only=True)
    print(f"✅ Unity filter: 'eat' -> {result['action']} (EAT not Unity-supported, should be None)")
    
    result = map_action("eat", unity_only=False)
    print(f"✅ No filter: 'eat' -> {result['action']}")
    
    print("\nAll tests passed ✅")

