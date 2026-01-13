"""Comprehensive test suite for all SOFAI mode combinations.

Tests all combinations of:
- S2 solver modes: fresh_start, continue_chat, best_attempt
- Validation methods: custom validator, LLM-as-Judge
- Feedback strategies: simple, first_error, all_errors
- Domain test cases: global, per-requirement
"""

import asyncio
import json
from typing import Literal

import mellea
from mellea.backends.ollama import OllamaModelBackend
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import ChatContext
from mellea.stdlib.requirement import req, Requirement, ValidationResult
from mellea.stdlib.sampling import SOFAISamplingStrategy


# Test problem: Simple JSON formatting task
INSTRUCTION = "Create a simple greeting message for a user named Alice. Output as JSON with 'name' and 'greeting' fields."

def parse_json(output_str: str) -> dict | None:
    """Parse LLM output as JSON."""
    try:
        output_str = output_str.strip()
        if output_str.startswith("```json"):
            output_str = output_str[7:].split("```")[0].strip()
        elif output_str.startswith("```"):
            output_str = output_str[3:].split("```")[0].strip()
        parsed = json.loads(output_str)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, Exception):
        return None


def create_custom_validator(check_fields: bool = True) -> callable:
    """Create a custom validator for JSON output."""
    def validate_json(ctx) -> ValidationResult:
        output = ctx.last_output()
        if output is None:
            return ValidationResult(False, reason="No output found")
        
        parsed = parse_json(str(output.value))
        if parsed is None:
            return ValidationResult(False, reason="Invalid JSON format")
        
        if check_fields:
            if "name" not in parsed:
                return ValidationResult(False, reason="Missing 'name' field")
            if "greeting" not in parsed:
                return ValidationResult(False, reason="Missing 'greeting' field")
            if parsed.get("name") != "Alice":
                return ValidationResult(False, reason="Name should be 'Alice'")
        
        return ValidationResult(True, reason="Valid output")
    
    return validate_json


DOMAIN_TEST_CASE_GLOBAL = """
Test case: The output must be valid JSON with the following structure:
{
  "name": "Alice",
  "greeting": "<a friendly greeting for Alice>"
}
"""

DOMAIN_TEST_CASE_PER_REQ = """
Specific test: Verify that the JSON contains exactly two fields: 'name' and 'greeting'.
The 'name' field must contain the string "Alice".
"""


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = []
        self.failed = []
    
    def add_pass(self, test_name: str):
        self.passed.append(test_name)
        print(f"  ✓ {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        print(f"  ✗ {test_name}: {error}")
    
    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}")
        print(f"Test Results: {len(self.passed)}/{total} passed")
        print(f"{'='*60}")
        if self.failed:
            print("\nFailed tests:")
            for name, error in self.failed:
                print(f"  ✗ {name}")
                print(f"    Error: {error}")
        return len(self.failed) == 0


async def test_s2_mode(
    mode: Literal["fresh_start", "continue_chat", "best_attempt"],
    results: TestResults
):
    """Test a specific S2 solver mode with custom validator."""
    test_name = f"S2 Mode: {mode}"
    try:
        s1 = OllamaModelBackend(model_id="llama3.2:1b")
        s2 = OllamaModelBackend(model_id="llama3.2:1b")
        
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=s1,
            s2_solver_backend=s2,
            s2_solver_mode=mode,
            loop_budget=2,
        )
        
        requirements = [req(
            description="Output must be valid JSON with 'name' and 'greeting' fields",
            validation_fn=create_custom_validator()
        )]
        
        m = mellea.MelleaSession(backend=s1, ctx=ChatContext())
        
        result = m.instruct(
            INSTRUCTION,
            requirements=requirements,
            strategy=strategy,
            return_sampling_results=True,
            model_options={"temperature": 0.1},
        )
        
        if result is not None:
            results.add_pass(test_name)
        else:
            results.add_fail(test_name, "Result is None")
            
    except Exception as e:
        results.add_fail(test_name, str(e))


async def test_llm_judge_feedback(
    feedback_strategy: Literal["simple", "first_error", "all_errors"],
    results: TestResults
):
    """Test LLM-as-Judge with different feedback strategies."""
    test_name = f"LLM-as-Judge: {feedback_strategy}"
    try:
        s1 = OllamaModelBackend(model_id="llama3.2:1b")
        s2 = OllamaModelBackend(model_id="llama3.2:1b")
        judge = OllamaModelBackend(model_id="llama3.2:1b")
        
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=s1,
            s2_solver_backend=s2,
            judge_backend=judge,
            feedback_strategy=feedback_strategy,
            loop_budget=2,
        )
        
        requirements = [
            Requirement(description="Output must be valid JSON"),
            Requirement(description="JSON must contain 'name' field with value 'Alice'"),
        ]
        
        m = mellea.MelleaSession(backend=s1, ctx=ChatContext())
        
        result = m.instruct(
            INSTRUCTION,
            requirements=requirements,
            strategy=strategy,
            return_sampling_results=True,
            model_options={"temperature": 0.1},
        )
        
        if result is not None:
            results.add_pass(test_name)
        else:
            results.add_fail(test_name, "Result is None")
            
    except Exception as e:
        results.add_fail(test_name, str(e))


async def test_domain_testcase_global(results: TestResults):
    """Test global domain test case."""
    test_name = "Domain Test Case: Global"
    try:
        s1 = OllamaModelBackend(model_id="llama3.2:1b")
        s2 = OllamaModelBackend(model_id="llama3.2:1b")
        judge = OllamaModelBackend(model_id="llama3.2:1b")
        
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=s1,
            s2_solver_backend=s2,
            judge_backend=judge,
            feedback_strategy="all_errors",
            domain_testcase=DOMAIN_TEST_CASE_GLOBAL,
            loop_budget=2,
        )
        
        requirements = [
            Requirement(description="Output must be valid JSON with correct structure"),
        ]
        
        m = mellea.MelleaSession(backend=s1, ctx=ChatContext())
        
        result = m.instruct(
            INSTRUCTION,
            requirements=requirements,
            strategy=strategy,
            return_sampling_results=True,
            model_options={"temperature": 0.1},
        )
        
        if result is not None:
            results.add_pass(test_name)
        else:
            results.add_fail(test_name, "Result is None")
            
    except Exception as e:
        results.add_fail(test_name, str(e))


async def test_domain_testcase_per_requirement(results: TestResults):
    """Test per-requirement domain test case."""
    test_name = "Domain Test Case: Per-Requirement"
    try:
        s1 = OllamaModelBackend(model_id="llama3.2:1b")
        s2 = OllamaModelBackend(model_id="llama3.2:1b")
        judge = OllamaModelBackend(model_id="llama3.2:1b")
        
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=s1,
            s2_solver_backend=s2,
            judge_backend=judge,
            feedback_strategy="first_error",
            loop_budget=2,
        )
        
        # Create requirement with domain_testcase
        req_with_testcase = Requirement(
            description="Output must have correct fields",
            domain_testcase=DOMAIN_TEST_CASE_PER_REQ
        )
        
        requirements = [req_with_testcase]
        
        m = mellea.MelleaSession(backend=s1, ctx=ChatContext())
        
        result = m.instruct(
            INSTRUCTION,
            requirements=requirements,
            strategy=strategy,
            return_sampling_results=True,
            model_options={"temperature": 0.1},
        )
        
        if result is not None:
            results.add_pass(test_name)
        else:
            results.add_fail(test_name, "Result is None")
            
    except Exception as e:
        results.add_fail(test_name, str(e))


async def test_combined_modes(results: TestResults):
    """Test combination of S2 modes with LLM-as-Judge."""
    for s2_mode in ["fresh_start", "continue_chat", "best_attempt"]:
        test_name = f"Combined: {s2_mode} + LLM-as-Judge (all_errors)"
        try:
            s1 = OllamaModelBackend(model_id="llama3.2:1b")
            s2 = OllamaModelBackend(model_id="llama3.2:1b")
            judge = OllamaModelBackend(model_id="llama3.2:1b")
            
            strategy = SOFAISamplingStrategy(
                s1_solver_backend=s1,
                s2_solver_backend=s2,
                s2_solver_mode=s2_mode,
                judge_backend=judge,
                feedback_strategy="all_errors",
                domain_testcase=DOMAIN_TEST_CASE_GLOBAL,
                loop_budget=2,
            )
            
            requirements = [
                Requirement(description="Output must be valid JSON"),
            ]
            
            m = mellea.MelleaSession(backend=s1, ctx=ChatContext())
            
            result = m.instruct(
                INSTRUCTION,
                requirements=requirements,
                strategy=strategy,
                return_sampling_results=True,
                model_options={"temperature": 0.1},
            )
            
            if result is not None:
                results.add_pass(test_name)
            else:
                results.add_fail(test_name, "Result is None")
                
        except Exception as e:
            results.add_fail(test_name, str(e))


async def main():
    """Run all tests."""
    print("="*60)
    print("SOFAI Mode Combinations - Comprehensive Test Suite")
    print("="*60)
    
    results = TestResults()
    
    # Test S2 modes with custom validator
    print("\n[1/6] Testing S2 Solver Modes...")
    for mode in ["fresh_start", "continue_chat", "best_attempt"]:
        await test_s2_mode(mode, results)
    
    # Test LLM-as-Judge with different feedback strategies
    print("\n[2/6] Testing LLM-as-Judge Feedback Strategies...")
    for strategy in ["simple", "first_error", "all_errors"]:
        await test_llm_judge_feedback(strategy, results)
    
    # Test domain test cases
    print("\n[3/6] Testing Domain Test Cases...")
    await test_domain_testcase_global(results)
    await test_domain_testcase_per_requirement(results)
    
    # Test combined modes
    print("\n[4/6] Testing Combined Modes...")
    await test_combined_modes(results)
    
    # Show summary
    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    import logging
    FancyLogger.get_logger().setLevel(logging.WARNING)  # Reduce noise
    exit(asyncio.run(main()))
