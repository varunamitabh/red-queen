"""lerobot-eval, optionally with LIBERO-plus file-name suffixes removed from the instruction.

  python scripts/hpc/run_eval.py [--clean-language] <any lerobot-eval arguments>

With --clean-language the instruction table is patched in this process, so environments
are forced to run in-process (--eval.use_async_envs=false); otherwise worker processes
could rebuild the unpatched table.
"""
import sys
from importlib.metadata import entry_points

args = sys.argv[1:]
if "--clean-language" in args:
    args.remove("--clean-language")
    import libero.libero.benchmark as bench

    from redqueen.language import clean_instruction

    changed = 0
    for tasks in bench.task_maps.values():
        for name, task in list(tasks.items()):
            text = clean_instruction(name)
            if text is not None and text != task.language:
                tasks[name] = task._replace(language=text)
                changed += 1
    print(f"clean-language: rewrote {changed} instructions", flush=True)
    args = [a for a in args if not a.startswith("--eval.use_async_envs")] + ["--eval.use_async_envs=false"]

(main,) = [e for e in entry_points(group="console_scripts") if e.name == "lerobot-eval"]
sys.argv = ["lerobot-eval", *args]
sys.exit(main.load()())