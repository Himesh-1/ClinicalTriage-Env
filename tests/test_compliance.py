"""
tests/test_compliance.py — Pre-submission validation tests for ClinicalTriage-Env

Run with:
  pytest tests/test_compliance.py -v

Or as a standalone script:
  python tests/test_compliance.py

Both require the server to be running on http://localhost:7860
"""

import json
import os
import sys
import httpx
import pytest

BASE_URL = "http://localhost:7860"
CLIENT = httpx.Client(base_url=BASE_URL, timeout=30)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────────────────────────────────────

def post_step(action: dict, session_id: str = "test") -> httpx.Response:
    return CLIENT.post("/step", json={"action": action, "session_id": session_id})


def post_reset(task_name: str, session_id: str = "test") -> httpx.Response:
    return CLIENT.post("/reset", json={"task_name": task_name, "session_id": session_id})


# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self):
        resp = CLIENT.get("/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

    def test_health_returns_ok(self):
        resp = CLIENT.get("/health")
        data = resp.json()
        assert data.get("status") == "ok", f"Expected 'ok', got {data}"


class TestTasks:
    def test_tasks_endpoint_exists(self):
        resp = CLIENT.get("/tasks")
        assert resp.status_code == 200

    def test_tasks_returns_list(self):
        resp = CLIENT.get("/tasks")
        data = resp.json()
        assert isinstance(data, list), "Expected list of tasks"
        assert len(data) >= 3, "Expected at least 3 tasks"

    def test_all_three_tasks_present(self):
        resp = CLIENT.get("/tasks")
        names = [t["name"] for t in resp.json()]
        assert "single_patient_easy" in names
        assert "concurrent_patients_medium" in names
        assert "incomplete_vitals_hard" in names


class TestReset:
    def test_reset_returns_200(self):
        resp = post_reset("single_patient_easy", "cr1")
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"

    def test_reset_returns_patient_observation(self):
        resp = post_reset("single_patient_easy", "cr2")
        data = resp.json()
        required_fields = [
            "patient_id", "age", "heart_rate", "blood_pressure_systolic",
            "blood_pressure_diastolic", "oxygen_saturation", "respiratory_rate",
            "temperature", "pain_scale", "chief_complaint", "arrival_mode",
            "vitals_complete",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_reset_all_tasks(self):
        for task in ["single_patient_easy", "concurrent_patients_medium", "incomplete_vitals_hard"]:
            resp = post_reset(task, f"cr3_{task}")
            assert resp.status_code == 200, f"Reset failed for task {task}: {resp.text}"

    def test_reset_invalid_task_returns_400(self):
        resp = post_reset("nonexistent_task", "cr4")
        assert resp.status_code == 400

    def test_reset_cycles_patients(self):
        """Verify that consecutive resets return different patients."""
        resp1 = post_reset("single_patient_easy", "cycle1")
        resp2 = post_reset("single_patient_easy", "cycle1")
        p1 = resp1.json()["patient_id"]
        p2 = resp2.json()["patient_id"]
        assert p1 != p2, f"Expected different patients, got {p1} both times"


class TestStep:
    def setup_method(self):
        post_reset("single_patient_easy", "step_test")

    def test_step_returns_200(self):
        action = {"triage_level": 2, "reasoning": "High risk patient with severe vitals.", "confidence": 0.8}
        resp = post_step(action, "step_test")
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"

    def test_step_returns_step_result_fields(self):
        post_reset("single_patient_easy", "step_test2")
        action = {"triage_level": 1, "reasoning": "Critical life-threatening signs.", "confidence": 0.9}
        resp = post_step(action, "step_test2")
        data = resp.json()
        assert "reward" in data
        assert "done" in data
        assert "info" in data

    def test_step_reward_in_range(self):
        post_reset("single_patient_easy", "step_range")
        action = {"triage_level": 3, "reasoning": "Moderate symptoms, stable vitals.", "confidence": 0.7}
        resp = post_step(action, "step_range")
        data = resp.json()
        reward = data["reward"]
        assert 0.0 <= reward <= 1.0, f"Reward out of range: {reward}"

    def test_step_done_true_after_easy(self):
        post_reset("single_patient_easy", "step_done")
        action = {"triage_level": 1, "reasoning": "Critical.", "confidence": 1.0}
        resp = post_step(action, "step_done")
        data = resp.json()
        assert data["done"] is True

    def test_step_after_done_returns_400(self):
        post_reset("single_patient_easy", "step_400")
        action = {"triage_level": 1, "reasoning": "Critical.", "confidence": 1.0}
        post_step(action, "step_400")
        # Second step should fail
        resp = post_step(action, "step_400")
        assert resp.status_code == 400


class TestState:
    def test_state_returns_200(self):
        resp = CLIENT.get("/state", params={"session_id": "state_test"})
        assert resp.status_code == 200

    def test_state_returns_environment_state_fields(self):
        post_reset("single_patient_easy", "state_fields")
        resp = CLIENT.get("/state", params={"session_id": "state_fields"})
        data = resp.json()
        required = ["task_name", "step_count", "current_patients", "done", "cumulative_reward"]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_state_cumulative_reward_increases(self):
        post_reset("single_patient_easy", "state_cum")
        action = {"triage_level": 1, "reasoning": "Critical.", "confidence": 0.9}
        post_step(action, "state_cum")
        resp = CLIENT.get("/state", params={"session_id": "state_cum"})
        data = resp.json()
        assert data["cumulative_reward"] >= 0.0


class TestRewardValues:
    def test_exact_match_gives_higher_reward(self):
        """ESI-1 patient — correct answer should score higher than wrong answer"""
        post_reset("single_patient_easy", "reward_exact")
        # Get the current patient and try correct ESI-1
        correct = {
            "triage_level": 1,
            "reasoning": "Critical patient: severe respiratory distress, low SpO2, "
                         "hypotension, tachycardia — ESI-1 immediate intervention required. "
                         "Life-threatening vital signs mandate immediate resuscitation.",
            "confidence": 0.95,
        }
        resp = post_step(correct, "reward_exact")
        exact_reward = resp.json()["reward"]

        post_reset("single_patient_easy", "reward_wrong")
        wrong = {"triage_level": 5, "reasoning": "Non-urgent patient.", "confidence": 0.3}
        resp = post_step(wrong, "reward_wrong")
        wrong_reward = resp.json()["reward"]

        assert exact_reward > wrong_reward, (
            f"Exact match reward ({exact_reward}) should be > wrong reward ({wrong_reward})"
        )


class TestMediumTask:
    def test_medium_reset_returns_patient(self):
        resp = post_reset("concurrent_patients_medium", "med_reset")
        assert resp.status_code == 200
        data = resp.json()
        assert "patient_id" in data

    def test_medium_state_has_5_patients(self):
        post_reset("concurrent_patients_medium", "med_state")
        resp = CLIENT.get("/state", params={"session_id": "med_state"})
        data = resp.json()
        assert len(data["current_patients"]) == 5

    def test_medium_step_with_json_ranking(self):
        post_reset("concurrent_patients_medium", "med_json")
        state = CLIENT.get("/state", params={"session_id": "med_json"}).json()
        # Extract patient IDs in order they appear
        pids = [p["patient_id"] for p in state["current_patients"]]
        action = {
            "triage_level": 1,
            "reasoning": json.dumps(pids),  # Submit as JSON ranking
            "confidence": 0.85,
        }
        resp = post_step(action, "med_json")
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["reward"] <= 1.0


class TestHardTask:
    def test_hard_request_vitals_accepts_level_0(self):
        """Verifies BUG 1 fix — triage_level=0 no longer rejected."""
        post_reset("incomplete_vitals_hard", "hard_0")
        action = {"triage_level": 0, "reasoning": "Requesting complete vitals before triaging.", "confidence": 0.5}
        resp = post_step(action, "hard_0")
        assert resp.status_code == 200, f"triage_level=0 was rejected: {resp.status_code} {resp.text}"

    def test_hard_request_vitals_episode_continues(self):
        post_reset("incomplete_vitals_hard", "hard1")
        action = {"triage_level": 0, "reasoning": "Requesting complete vitals.", "confidence": 0.5}
        resp = post_step(action, "hard1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is False  # Episode continues after requesting vitals
        assert data["reward"] == 0.0

    def test_hard_vitals_revealed_after_request(self):
        post_reset("incomplete_vitals_hard", "hard_reveal")
        action = {"triage_level": 0, "reasoning": "Requesting vitals.", "confidence": 0.5}
        resp = post_step(action, "hard_reveal")
        data = resp.json()
        obs = data["observation"]
        assert obs["vitals_complete"] is True, "Vitals should be complete after reveal"

    def test_hard_final_triage_returns_reward(self):
        post_reset("incomplete_vitals_hard", "hard2")
        # Step 1: request vitals
        action = {"triage_level": 0, "reasoning": "Requesting vitals.", "confidence": 0.5}
        post_step(action, "hard2")
        # Step 2: final triage
        action2 = {
            "triage_level": 2,
            "reasoning": "Chest pain patient with beta-blockers, elevated BP, SpO2 93% - ESI-2 emergent.",
            "confidence": 0.8,
        }
        resp = post_step(action2, "hard2")
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["reward"] <= 1.0
        assert data["done"] is True


class TestOpenEnvYaml:
    def test_yaml_file_exists(self):
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "openenv.yaml")
        assert os.path.exists(yaml_path), "openenv.yaml not found in root"

    def test_yaml_has_required_fields(self):
        import yaml
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "openenv.yaml")
        with open(yaml_path, encoding='utf-8') as f:
            data = yaml.safe_load(f)
        assert "name" in data
        assert "version" in data
        assert "tasks" in data
        assert len(data["tasks"]) >= 3
        assert "action_space" in data
        assert "observation_space" in data
        assert "reward_range" in data

    def test_yaml_task_names_match(self):
        import yaml
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "openenv.yaml")
        with open(yaml_path, encoding='utf-8') as f:
            data = yaml.safe_load(f)
        task_names = [t["name"] for t in data["tasks"]]
        assert "single_patient_easy" in task_names
        assert "concurrent_patients_medium" in task_names
        assert "incomplete_vitals_hard" in task_names


class TestInferenceFileCompliance:
    """Verify inference.py meets spec requirements (static checks)."""

    def test_inference_exists_in_root(self):
        inf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inference.py")
        assert os.path.exists(inf_path), "inference.py must be in root directory"

    def test_inference_uses_openai_client(self):
        inf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inference.py")
        with open(inf_path, encoding='utf-8') as f:
            content = f.read()
        assert "from openai import OpenAI" in content, "Must import OpenAI client"

    def test_inference_uses_required_env_vars(self):
        inf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inference.py")
        with open(inf_path, encoding='utf-8') as f:
            content = f.read()
        assert "API_BASE_URL" in content, "Must use API_BASE_URL env var"
        assert "MODEL_NAME" in content, "Must use MODEL_NAME env var"
        assert "HF_TOKEN" in content, "Must use HF_TOKEN env var"

    def test_inference_has_correct_log_format(self):
        inf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inference.py")
        with open(inf_path, encoding='utf-8') as f:
            content = f.read()
        assert "[START]" in content, "Must have [START] log"
        assert "[STEP]" in content, "Must have [STEP] log"
        assert "[END]" in content, "Must have [END] log"


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  ClinicalTriage-Env — Pre-Submission Compliance Check")
    print("=" * 65)

    test_classes = [
        TestHealth, TestTasks, TestReset, TestStep,
        TestState, TestRewardValues, TestMediumTask, TestHardTask,
        TestOpenEnvYaml, TestInferenceFileCompliance,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        print(f"\n  [{cls.__name__}]")
        instance = cls()
        methods = sorted([m for m in dir(instance) if m.startswith("test_")])
        for method_name in methods:
            method = getattr(instance, method_name)
            if hasattr(instance, "setup_method"):
                try:
                    instance.setup_method()
                except Exception:
                    pass
            try:
                method()
                print(f"    [PASS] {method_name}")
                passed += 1
            except AssertionError as e:
                print(f"    [FAIL] {method_name}: {e}")
                errors.append(f"{cls.__name__}.{method_name}: {e}")
                failed += 1
            except Exception as e:
                print(f"    [ERR]  {method_name}: ERROR - {e}")
                errors.append(f"{cls.__name__}.{method_name}: ERROR - {e}")
                failed += 1

    print()
    print("-" * 65)
    print(f"  Results: {passed} passed, {failed} failed")
    if errors:
        print("\n  Failures:")
        for err in errors:
            print(f"    - {err}")
        sys.exit(1)
    else:
        print("  All checks passed!")
        sys.exit(0)
