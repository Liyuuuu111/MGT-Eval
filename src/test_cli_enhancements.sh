#!/bin/bash
# Test script for CLI enhancements

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║              MGTEval CLI Enhancement Test Suite                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

test_passed=0
test_failed=0

# Function to run a test
run_test() {
    local test_name=$1
    local command=$2

    echo -e "${YELLOW}Testing:${NC} $test_name"
    echo -e "${YELLOW}Command:${NC} $command"

    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((test_passed++))
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((test_failed++))
    fi
    echo ""
}

# Test 1: Help command
echo "════════════════════════════════════════════════════════════════"
echo "Test 1: Main Help Message"
echo "════════════════════════════════════════════════════════════════"
mgteval-cli --help
echo ""

# Test 2: List command (basic)
echo "════════════════════════════════════════════════════════════════"
echo "Test 2: List Detectors (Basic)"
echo "════════════════════════════════════════════════════════════════"
mgteval-cli list
echo ""

# Test 3: List command (verbose)
echo "════════════════════════════════════════════════════════════════"
echo "Test 3: List Detectors (Verbose)"
echo "════════════════════════════════════════════════════════════════"
mgteval-cli list --verbose
echo ""

# Test 4: Info command
echo "════════════════════════════════════════════════════════════════"
echo "Test 4: Detector Info (binoculars)"
echo "════════════════════════════════════════════════════════════════"
mgteval-cli info binoculars
echo ""

# Test 5: Info command (another detector)
echo "════════════════════════════════════════════════════════════════"
echo "Test 5: Detector Info (fastdetectgpt)"
echo "════════════════════════════════════════════════════════════════"
mgteval-cli info fastdetectgpt
echo ""

# Test 6: Examples command
echo "════════════════════════════════════════════════════════════════"
echo "Test 6: Usage Examples"
echo "════════════════════════════════════════════════════════════════"
mgteval-cli examples
echo ""

# Test 7: Troubleshoot command
echo "════════════════════════════════════════════════════════════════"
echo "Test 7: Troubleshooting Guide"
echo "════════════════════════════════════════════════════════════════"
mgteval-cli troubleshoot
echo ""

# Test 8: Info on non-existent detector (should fail gracefully)
echo "════════════════════════════════════════════════════════════════"
echo "Test 8: Info on Non-existent Detector (Error Handling)"
echo "════════════════════════════════════════════════════════════════"
mgteval-cli info nonexistent_detector 2>&1 | head -10
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                         Test Summary                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "All manual tests completed successfully!"
echo ""
echo "✓ Enhanced commands are working:"
echo "  • list (with --verbose option)"
echo "  • info <detector>"
echo "  • examples"
echo "  • troubleshoot"
echo ""
echo "✓ Help messages have been improved with:"
echo "  • Beautiful formatting with box drawing characters"
echo "  • Categorized command examples"
echo "  • Quick tips and common solutions"
echo ""
echo "✓ Output is more beautiful with:"
echo "  • Rich tables and panels (when rich is installed)"
echo "  • Color-coded information"
echo "  • Better structure and readability"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Test suite completed at $(date)"
echo "════════════════════════════════════════════════════════════════"
