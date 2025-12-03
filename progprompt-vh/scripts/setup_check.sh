#!/bin/bash

# Setup check script for agent integration with progprompt-vh
# This script verifies that all required components are properly configured

echo "========================================================================"
echo "Agent Integration Setup Check"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check 1: Agent directory exists
echo "Checking for agent directory..."
if [ -d "../../agent" ]; then
    check_pass "Agent directory found"
else
    check_fail "Agent directory not found at ../../agent"
    echo "  Please ensure agent/ is in the correct location"
fi
echo ""

# Check 2: VirtualHome directory exists
echo "Checking for VirtualHome..."
if [ -d "../../virtualhome" ]; then
    check_pass "VirtualHome directory found"
else
    check_fail "VirtualHome directory not found at ../../virtualhome"
    echo "  Clone it with: git clone https://github.com/xavierpuigf/virtualhome.git"
fi
echo ""

# Check 3: Python can import agent modules
echo "Checking if agent modules are importable..."
python3 -c "import sys; sys.path.insert(0, '../../agent'); import planner" 2>/dev/null
if [ $? -eq 0 ]; then
    check_pass "Agent modules can be imported"
else
    check_warn "Cannot import agent modules (may need to set PYTHONPATH)"
    echo "  Run: export PYTHONPATH=\$(pwd)/../../agent:\$PYTHONPATH"
fi
echo ""

# Check 4: Python can import VirtualHome
echo "Checking if VirtualHome is importable..."
python3 -c "from virtualhome.simulation.unity_simulator import comm_unity" 2>/dev/null
if [ $? -eq 0 ]; then
    check_pass "VirtualHome can be imported"
else
    check_warn "Cannot import VirtualHome (may need to install it)"
    echo "  Run: cd ../../virtualhome && pip install -e ."
fi
echo ""

# Check 5: OpenAI API key
echo "Checking environment variables..."
if [ -n "$OPENAI_API_KEY" ]; then
    check_pass "OPENAI_API_KEY is set"
else
    check_warn "OPENAI_API_KEY not set"
    echo "  Run: export OPENAI_API_KEY=your-api-key"
fi

if [ -n "$OPENAI_BASE_URL" ]; then
    check_pass "OPENAI_BASE_URL is set"
else
    check_warn "OPENAI_BASE_URL not set (will use default)"
    echo "  Run: export OPENAI_BASE_URL=https://api.openai.com/v1"
fi
echo ""

# Check 6: Unity executable
echo "Checking for Unity executable..."
if [ -n "$VH_EXECUTABLE" ]; then
    if [ -e "$VH_EXECUTABLE" ]; then
        check_pass "VH_EXECUTABLE is set and file exists: $VH_EXECUTABLE"
    else
        check_fail "VH_EXECUTABLE is set but file not found: $VH_EXECUTABLE"
    fi
else
    check_warn "VH_EXECUTABLE not set"
    echo "  Download from: https://github.com/xavierpuigf/virtualhome/releases"
    echo "  Then run: export VH_EXECUTABLE=/path/to/macos_exec.v2.3.0.app"
fi
echo ""

# Check 7: Required Python packages
echo "Checking Python packages..."
python3 -c "import openai" 2>/dev/null
if [ $? -eq 0 ]; then
    check_pass "openai package installed"
else
    check_fail "openai package not installed"
    echo "  Run: pip install openai"
fi

python3 -c "import cv2" 2>/dev/null
if [ $? -eq 0 ]; then
    check_pass "opencv-python package installed"
else
    check_warn "opencv-python not installed (needed for video recording)"
    echo "  Run: pip install opencv-python"
fi
echo ""

# Check 8: Integration scripts exist
echo "Checking integration scripts..."
if [ -f "agent_adapter.py" ]; then
    check_pass "agent_adapter.py found"
else
    check_fail "agent_adapter.py not found"
fi

if [ -f "run_agent_eval.py" ]; then
    check_pass "run_agent_eval.py found"
else
    check_fail "run_agent_eval.py not found"
fi

if [ -f "run_agent_eval_limited.py" ]; then
    check_pass "run_agent_eval_limited.py found"
else
    check_fail "run_agent_eval_limited.py not found"
fi
echo ""

# Summary
echo "========================================================================"
echo "Setup Check Complete"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Fix any failed checks above"
echo "2. Read AGENT_INTEGRATION_README.md for usage instructions"
echo "3. Run a quick test with run_agent_eval_limited.py"
echo ""
echo "Example command:"
echo "  python scripts/run_agent_eval_limited.py \\"
echo "    --progprompt-path \$(pwd) \\"
echo "    --expt-name quick_test \\"
echo "    --unity-filename \$VH_EXECUTABLE \\"
echo "    --test-set test_seen \\"
echo "    --num-instances 2 \\"
echo "    --agent-model gpt-4o"
echo ""
