"""Run model-produced code in short-lived, memory-bounded, network-restricted subprocesses."""
from __future__ import annotations

import builtins
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
TIME_LIMIT_SECONDS = 3.0
_FENCE = re.compile(r"^\s*```[\w+-]*\s*\n([\s\S]*?)\n```\s*$")
_PY_RUNNER = r'''import builtins, json, pathlib, sys
payload = json.loads(sys.stdin.read())
source = pathlib.Path("submission.py").read_text(encoding="utf-8")
allowed = {"math", "re", "collections", "itertools", "functools", "statistics", "datetime", "json", "decimal", "fractions", "string"}
blocked = {"socket", "subprocess", "multiprocessing", "ctypes", "os", "sys", "importlib", "urllib", "http", "ssl", "ftplib", "pathlib", "shutil", "tempfile", "resource"}
real_import = builtins.__import__
def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root in blocked or root not in allowed:
        raise ImportError("module is outside the offline evaluation allowlist")
    return real_import(name, globals, locals, fromlist, level)
safe_builtins = dict(builtins.__dict__)
for key in ("open", "eval", "exec", "compile", "input", "breakpoint", "help", "print"):
    safe_builtins.pop(key, None)
safe_builtins["__import__"] = safe_import
namespace = {"__builtins__": safe_builtins, "__name__": "submission"}
exec(compile(source, "submission.py", "exec"), namespace, namespace)
function = namespace[payload["function"]]
results = [function(*case[:-1]) for case in payload["tests"]]
output = json.dumps(results, ensure_ascii=False, allow_nan=False)
if len(output) > 1000000: raise SystemExit(3)
sys.stdout.write(output)
'''


def strip_code_fence(text: str) -> str:
    match = _FENCE.fullmatch(text)
    return match.group(1) if match else text.strip()


def _posix_limits() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (int(TIME_LIMIT_SECONDS) + 1, int(TIME_LIMIT_SECONDS) + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024, 2 * 1024 * 1024))


def _run_process(command: list[str], cwd: str, input_text: str, timeout: float = TIME_LIMIT_SECONDS) -> tuple[int, str]:
    kwargs: dict[str, Any] = {
        "cwd": cwd, "stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL,
        "text": True, "encoding": "utf-8", "errors": "replace", "env": {"PATH": os.environ.get("PATH", "")},
    }
    if os.name == "posix":
        kwargs["preexec_fn"] = _posix_limits
    elif os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process: subprocess.Popen | None = None
    try:
        process = subprocess.Popen(command, **kwargs)
        job_handle = _assign_windows_job(process) if os.name == "nt" else None
        try:
            output, _ = process.communicate(input_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return 124, ""
        finally:
            if job_handle is not None:
                import ctypes
                ctypes.windll.kernel32.CloseHandle(job_handle)
    except (OSError, RuntimeError):
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        return 124, ""
    return process.returncode, output[:100_000]


def _assign_windows_job(process: subprocess.Popen) -> int:
    """Apply a per-process memory cap so a generated snippet cannot exhaust the host."""
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel = ctypes.windll.kernel32
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise RuntimeError("Could not create a resource-limited evaluation job.")
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = 0x0100 | 0x2000  # process memory cap and kill on job close
    limits.ProcessMemoryLimit = MEMORY_LIMIT_BYTES
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        kernel.CloseHandle(job)
        raise RuntimeError("Could not set evaluation process limits.")
    if not kernel.AssignProcessToJobObject(job, wintypes.HANDLE(int(process._handle))):
        kernel.CloseHandle(job)
        process.kill()
        raise RuntimeError("Could not apply evaluation process limits.")
    return int(job)


def run_python_tests(source: str, function: str, tests: list[list[Any]]) -> tuple[bool, str]:
    try:
        with tempfile.TemporaryDirectory(prefix="medha-eval-") as directory:
            Path(directory, "submission.py").write_text(strip_code_fence(source), encoding="utf-8")
            payload = json.dumps({"function": function, "tests": tests}, ensure_ascii=False)
            Path(directory, "runner.py").write_text(_PY_RUNNER, encoding="utf-8")
            code, output = _run_process([sys.executable, "-I", "-S", "runner.py"], directory, payload)
        if code != 0:
            return False, "python process failed, timed out, or requested a blocked capability"
        actual = json.loads(output)
        expected = [case[-1] for case in tests]
        return (actual == expected, "" if actual == expected else "hidden Python cases did not match")
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False, "invalid Python task or output"


def run_javascript_tests(source: str, function: str, tests: list[list[Any]]) -> tuple[bool, str]:
    payload = json.dumps(tests, ensure_ascii=False).replace("\\u2028", "\\\\u2028").replace("\\u2029", "\\\\u2029")
    runner = f'''const vm=require("vm"), fs=require("fs");
const testData={payload};
try {{ const code=fs.readFileSync("submission.js", "utf8") + "\\nconst cases=JSON.parse(" + JSON.stringify(JSON.stringify(testData)) + "); JSON.stringify(cases.map(c=>{function}(...c.slice(0,-1))))";
const result=String(vm.runInNewContext(code, Object.create(null), {{timeout: 700, codeGeneration: {{strings: false, wasm: false}}}})); if(result.length>1000000) process.exit(3); process.stdout.write(result); }} catch (e) {{ process.exit(2); }}
'''
    try:
        with tempfile.TemporaryDirectory(prefix="medha-eval-js-") as directory:
            Path(directory, "submission.js").write_text(strip_code_fence(source), encoding="utf-8")
            Path(directory, "runner.js").write_text(runner, encoding="utf-8")
            code, output = _run_process(["node", "runner.js"], directory, "", timeout=TIME_LIMIT_SECONDS)
        if code != 0:
            return False, f"JavaScript subprocess failed with status {code} (timeout, resource limit, or blocked capability)"
        actual = json.loads(output.strip())
        expected = [case[-1] for case in tests]
        return (actual == expected, "" if actual == expected else "hidden JavaScript cases did not match")
    except (OSError, json.JSONDecodeError, TypeError):
        return False, "invalid JavaScript task or output"


def run_sql_tests(query: str, tables: list, expected_rows: list) -> tuple[bool, str]:
    query = strip_code_fence(query).rstrip().rstrip(";")
    runner = r'''import json, sqlite3, sys
payload=json.loads(sys.stdin.read()); db=sqlite3.connect(":memory:"); db.enable_load_extension(False)
try:
 for name, columns, rows in payload["tables"]:
  db.execute("CREATE TABLE " + name + " (" + columns + ")")
  if rows: db.executemany("INSERT INTO " + name + " VALUES (" + ",".join("?" for _ in rows[0]) + ")", rows)
 allowed={sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
 db.set_authorizer(lambda action,*args: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
 db.set_progress_handler(lambda: 1,100000)
 actual=db.execute(payload["query"]).fetchall()
 sys.stdout.write(json.dumps(actual == [tuple(row) for row in payload["expected"]]))
except Exception: sys.stdout.write("false")
finally: db.close()
'''
    try:
        with tempfile.TemporaryDirectory(prefix="medha-eval-sql-") as directory:
            Path(directory, "runner.py").write_text(runner, encoding="utf-8")
            payload = json.dumps({"query": query, "tables": tables, "expected": expected_rows})
            code, output = _run_process([sys.executable, "-I", "-S", "runner.py"], directory, payload)
        passed = code == 0 and json.loads(output) is True
        return passed, "" if passed else "SQL fixture failed, timed out, or attempted a blocked database operation"
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False, "invalid SQL task or checker output"


def run_markup_tests(source: str, language: str, task: dict) -> tuple[bool, str]:
    source = strip_code_fence(source)
    runner = r'''import json,re,sys
from html.parser import HTMLParser
payload=json.loads(sys.stdin.read()); source=payload["source"]
if payload["language"] == "html":
 class Facts(HTMLParser):
  def __init__(self): super().__init__(convert_charrefs=True); self.tags=[]; self.attrs=[]; self.text=[]
  def handle_starttag(self,tag,attrs): self.tags.append(tag.lower()); self.attrs.extend((key.lower(),value or "") for key,value in attrs)
  def handle_data(self,data): self.text.append(data)
 facts=Facts()
 try: facts.feed(source); facts.close()
 except Exception: sys.stdout.write("false"); raise SystemExit
 checks=payload["task"]["checks"]
 attrs_ok=all(any(name==key and value.casefold()==expected.casefold() for name,value in facts.attrs) for key,expected in checks.get("attrs",{}).items())
 text=" ".join(facts.text).casefold()
 ok=all(tag in facts.tags for tag in checks.get("tags",[])) and attrs_ok and all(term.casefold() in text for term in checks.get("text",[]))
else:
 css=re.sub(r"/\*[\s\S]*?\*/","",source).casefold(); ok=True
 for selector,prop,value in payload["task"].get("rules",[]):
  if selector.startswith("@"):
   ok=ok and selector.casefold() in css and prop.casefold() in css and value.casefold() in css; continue
  blocks=re.findall(re.escape(selector.casefold())+r"\s*\{([^}]*)\}",css)
  ok=ok and bool(blocks) and any(re.search(r"(?:^|;)\s*"+re.escape(prop.casefold())+r"\s*:\s*[^;]*"+re.escape(value.casefold()),block) for block in blocks)
sys.stdout.write(json.dumps(bool(ok)))
'''
    try:
        with tempfile.TemporaryDirectory(prefix="medha-eval-markup-") as directory:
            Path(directory, "runner.py").write_text(runner, encoding="utf-8")
            payload = json.dumps({"source": source, "language": language, "task": task}, ensure_ascii=False)
            code, output = _run_process([sys.executable, "-I", "-S", "runner.py"], directory, payload)
        passed = code == 0 and json.loads(output) is True
        return passed, "" if passed else "markup checker failed, timed out, or structure was missing"
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False, "invalid markup task or checker output"


def run_code_tests(task: dict, answer: str) -> tuple[bool, str]:
    language = task.get("language")
    if language == "python":
        return run_python_tests(answer, task["function"], task["tests"])
    if language == "javascript":
        return run_javascript_tests(answer, task["function"], task["tests"])
    if language == "sql":
        return run_sql_tests(answer, task["setup"], task["expected_rows"])
    if language in {"html", "css"}:
        return run_markup_tests(answer, language, task)
    return False, "unsupported code language"
