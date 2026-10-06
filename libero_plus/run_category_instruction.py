"""Same as run_category.py, but can change the instruction given to the policy.
LP_INSTRUCTION=filename (default): LIBERO-Plus file-name text, with the metadata suffix
  ("... view 0 0 100 2 352 initstate 0"); this is what the published numbers used.
LP_INSTRUCTION=strip: the base task's file-name text without the suffix, identical to the
  clean LIBERO instruction the policies were trained on.
run_category.py is imported unchanged; the original OFT script is never edited."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_category as rc

MODE = os.environ.get("LP_INSTRUCTION", "filename")
_original_get_task = rc.CategorySubset.get_task

def get_task_strip(self, i):
    task = _original_get_task(self, i)
    base = task.bddl_file.split("_view_")[0].removesuffix(".bddl")
    language = " ".join(base.split("_"))
    print(f"[instruction] {task.language!r} -> {language!r}", flush=True)
    return task._replace(language=language)

if MODE == "strip":
    rc.CategorySubset.get_task = get_task_strip
elif MODE != "filename":
    raise ValueError(f"unknown LP_INSTRUCTION {MODE!r}")
print(f"[instruction] mode: {MODE}", flush=True)

if __name__ == "__main__":
    rc.rle.eval_libero()
