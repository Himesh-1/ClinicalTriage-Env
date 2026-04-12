import os
import re

file_path = 'd:/ClinicalTriage-Env/inference.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace Variables
content = content.replace(
    'API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")\nAPI_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")\nMODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")\n\nclient = OpenAI(\n    base_url=API_BASE_URL,\n    api_key=API_KEY,\n)',
    'API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")\nMODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")\nHF_TOKEN = os.getenv("HF_TOKEN")\n\nif HF_TOKEN is None:\n    raise ValueError("HF_TOKEN environment variable is required")\n\nclient = OpenAI(\n    base_url=API_BASE_URL,\n    api_key=HF_TOKEN,\n)'
)

# 2. log_end signature
content = content.replace(
    'def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:\n    rewards_str = ",".join(f"{r:.2f}" for r in rewards)\n    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)',
    'def log_end(success: bool, steps: int, rewards: List[float]) -> None:\n    rewards_str = ",".join(f"{r:.2f}" for r in rewards)\n    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)'
)

# 3. Runners error handling and remove score
def patch_runner(content, func_name):
    # Find the function body
    start_idx = content.find(f'def {func_name}(ep_num: int, session_id: str) -> tuple:')
    if start_idx == -1: return content
    
    # We will just replace the exact lines finding "success = total_reward > 0.0" 
    # to "return total_reward, step"
    
    old_block = """    success = total_reward > 0.0
    score = min(max(total_reward, 0.001), 0.999)
    log_end(success=success, steps=step, score=score, rewards=rewards)
    return total_reward, step"""
    
    new_block = """    success = total_reward > 0.0
    log_end(success=success, steps=step, rewards=rewards)
    return total_reward, step"""
    
    return content.replace(old_block, new_block)

content = patch_runner(content, "run_easy_episode")
content = patch_runner(content, "run_medium_episode")
content = patch_runner(content, "run_hard_episode")

# 4. Modify main exception handling
old_except = """            except Exception as exc:
                print(f"[ERROR] task={task_name} episode={ep} error={exc}")
                all_results[task_name].append({
                    "episode": ep,
                    "reward": 0.0,
                    "steps": 0,
                    "error": str(exc),
                })"""

new_except = """            except Exception as exc:
                import sys
                print(f"[ERROR] task={task_name} episode={ep} error={exc}", file=sys.stderr)
                log_end(success=False, steps=0, rewards=[])
                all_results[task_name].append({
                    "episode": ep,
                    "reward": 0.0,
                    "steps": 0,
                    "error": str(exc),
                })"""
content = content.replace(old_except, new_except)

# 5. Remove or stderr summary prints to avoid polluting stdout with non [XXX] lines
old_summary = """    elapsed = time.time() - start_time
    print()
    print("=" * 65)
    print("  ClinicalTriage-Env — Inference Summary")"""
new_summary = """    elapsed = time.time() - start_time
    import sys
    print(file=sys.stderr)
    print("=" * 65, file=sys.stderr)
    print("  ClinicalTriage-Env — Inference Summary", file=sys.stderr)"""

content = content.replace(old_summary, new_summary)
content = content.replace('print("=" * 65)', 'print("=" * 65, file=sys.stderr)')
content = content.replace('print(f"  Model:          {MODEL}")', 'print(f"  Model:          {MODEL}", file=sys.stderr)')
content = content.replace('print(f"  Total Time:     {elapsed:.1f}s ({elapsed/60:.1f} min)")', 'print(f"  Total Time:     {elapsed:.1f}s ({elapsed/60:.1f} min)", file=sys.stderr)')
content = content.replace('print(f"  Episodes/Task:  {EPISODES_PER_TASK}")', 'print(f"  Episodes/Task:  {EPISODES_PER_TASK}", file=sys.stderr)')
content = content.replace('print("-" * 65)', 'print("-" * 65, file=sys.stderr)')
content = content.replace('print(f"  {\'Task\':<35} {\'Avg Reward\':>10} {\'Episodes\':>10}")', 'print(f"  {\'Task\':<35} {\'Avg Reward\':>10} {\'Episodes\':>10}", file=sys.stderr)')
content = content.replace('print(f"  {task_name:<35} {avg:>10.4f} {len(results):>10}")', 'print(f"  {task_name:<35} {avg:>10.4f} {len(results):>10}", file=sys.stderr)')
content = content.replace('print(f"  {\'OVERALL AVERAGE\':<35} {overall_avg:>10.4f}")', 'print(f"  {\'OVERALL AVERAGE\':<35} {overall_avg:>10.4f}", file=sys.stderr)')
content = content.replace('print(f"\\n[INFO] Total inference time: {elapsed:.1f}s ({elapsed/60:.1f} min)")', 'print(f"\\n[INFO] Total inference time: {elapsed:.1f}s ({elapsed/60:.1f} min)", file=sys.stderr)')

# The above replacements correctly redirect all summary prints to stderr

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patching complete.")
