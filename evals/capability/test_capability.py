from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from backend.medha_brain import StubBrain
from evals.capability.bank import load_tasks
from evals.capability.checkers import check_answer
from evals.capability.loaders import load_public_jsonl
from evals.capability.runner import _wilson, run_suite
from evals.capability.sandbox import (
    run_javascript_tests,
    run_python_tests,
    run_sql_tests,
)


class CapabilityTests(unittest.TestCase):
    def test_bank_has_requested_original_category_counts(self):
        tasks = load_tasks()
        self.assertEqual(len(tasks), 210)
        counts = {
            category: sum(task["category"] == category for task in tasks)
            for category in {
                "knowledge_reasoning",
                "coding",
                "instruction_following",
                "multi_turn",
                "summarise_rewrite_translate",
                "safety_refusal",
            }
        }
        self.assertEqual(
            counts,
            {
                "knowledge_reasoning": 60,
                "coding": 40,
                "instruction_following": 40,
                "multi_turn": 20,
                "summarise_rewrite_translate": 30,
                "safety_refusal": 20,
            },
        )
        self.assertTrue(
            all(
                6 <= len(task["turns"]) <= 10
                for task in tasks
                if task["checker"] == "multi_turn"
            )
        )

    def test_constraints_check_format_word_count_and_json(self):
        task = {
            "checker": "constraints",
            "constraints": {"bullet_count": 2, "max_words_per_bullet": 5},
        }
        self.assertTrue(check_answer(task, "- Check the input\n- Save a result")[0])
        self.assertFalse(
            check_answer(task, "- Check the input\n- Save a result\n- Retry")[0]
        )
        json_task = {
            "checker": "constraints",
            "constraints": {
                "json_keys": ["name", "status"],
                "json_value_type": "string",
            },
        }
        self.assertTrue(check_answer(json_task, '{"name":"A","status":"ready"}')[0])
        self.assertFalse(check_answer(json_task, '{"name":"A","status":1}')[0])

    def test_python_hidden_tests_run_offline_in_a_subprocess(self):
        ok, _ = run_python_tests(
            "def normalize_spaces(value):\n    return ' '.join(value.split())\n",
            "normalize_spaces",
            [["  medha   ai  ", "medha ai"], ["one\\ntwo", "one\\ntwo"]],
        )
        self.assertTrue(ok)
        blocked, note = run_python_tests(
            "import socket\ndef normalize_spaces(value):\n    return value\n",
            "normalize_spaces",
            [["x", "x"]],
        )
        self.assertFalse(blocked)
        self.assertIn("blocked", note)

    def test_code_subprocess_memory_and_cpu_limits_are_enforced(self):
        import time

        started = time.monotonic()
        memory_ok, _ = run_python_tests(
            "def allocate(value):\n    large = bytearray(300 * 1024 * 1024)\n    return value\n",
            "allocate",
            [["bounded", "bounded"]],
        )
        self.assertFalse(memory_ok)
        self.assertLess(time.monotonic() - started, 2.0)
        started = time.monotonic()
        cpu_ok, _ = run_javascript_tests(
            "function loop(value) { while (true) {} }", "loop", [[1, 1]]
        )
        self.assertFalse(cpu_ok)
        self.assertLess(time.monotonic() - started, 2.0)

    @unittest.skipUnless(
        shutil.which("node"), "Node.js is needed for the offline JavaScript checker"
    )
    def test_javascript_hidden_cases_run_in_a_separate_context(self):
        ok, note = run_javascript_tests(
            "function uniqueInOrder(items) { return [...new Set(items)]; }",
            "uniqueInOrder",
            [[[3, 1, 3, 2], [3, 1, 2]], [[], []]],
        )
        self.assertTrue(ok, note)
        blocked, _ = run_javascript_tests(
            'function uniqueInOrder(items) { require("https"); return items; }',
            "uniqueInOrder",
            [[[1], [1]]],
        )
        self.assertFalse(blocked)

    def test_sql_checker_uses_read_only_select_and_fixture_database(self):
        tables = [
            (
                "learners",
                "id INTEGER, name TEXT, active INTEGER",
                [(1, "Ria", 1), (2, "Dev", 0), (3, "Asha", 1)],
            )
        ]
        ok, note = run_sql_tests(
            "SELECT name FROM learners WHERE active=1 ORDER BY name",
            tables,
            [("Asha",), ("Ria",)],
        )
        self.assertTrue(ok, note)
        blocked, _ = run_sql_tests("DELETE FROM learners", tables, [])
        self.assertFalse(blocked)

    def test_html_and_css_checkers_validate_required_structure(self):
        tasks = {
            task["id"]: task
            for task in load_tasks()
            if task.get("language") in {"html", "css"}
        }
        html_task = tasks["code-html-01"]
        self.assertTrue(
            check_answer(
                html_task,
                '<nav aria-label="Primary navigation"><a>Home</a><a>Research</a></nav>',
            )[0]
        )
        css_task = tasks["code-css-01"]
        self.assertTrue(
            check_answer(
                css_task, ".card { border: 1px solid #ccc; border-radius: 12px; }"
            )[0]
        )

    def test_all_forty_code_fixtures_accept_a_reference_solution(self):
        python = {
            "normalize_spaces": "def normalize_spaces(value): return ' '.join(value.split())",
            "unique_in_order": "def unique_in_order(values): return list(dict.fromkeys(values))",
            "running_totals": "def running_totals(values):\n total=0; result=[]\n for value in values: total+=value; result.append(total)\n return result",
            "clamp": "def clamp(value, low, high): return max(low, min(high, value))",
            "count_vowels": "def count_vowels(value): return sum(ch.casefold() in 'aeiou' for ch in value)",
            "rotate_left": "def rotate_left(values, k):\n if not values: return []\n k%=len(values); return values[k:]+values[:k]",
            "flatten_once": "def flatten_once(values): return [item for group in values for item in group]",
            "is_anagram": "def is_anagram(left, right): return sorted(left.replace(' ','').casefold()) == sorted(right.replace(' ','').casefold())",
            "stable_sort_records": "def stable_sort_records(records): return sorted(records, key=lambda row: row[1])",
            "safe_ratio": "def safe_ratio(numerator, denominator): return None if denominator == 0 else numerator / denominator",
        }
        javascript = {
            "uniqueInOrder": "function uniqueInOrder(values) { return [...new Set(values)]; }",
            "sumPositive": "function sumPositive(values) { return values.filter(x => x > 0).reduce((a,b) => a+b, 0); }",
            "titleWords": "function titleWords(value) { return value.split(' ').map(w => w[0].toUpperCase()+w.slice(1).toLowerCase()).join(' '); }",
            "rotateRight": "function rotateRight(values,k) { if(!values.length)return []; k%=values.length; return values.slice(-k).concat(values.slice(0,-k)); }",
            "countBy": "function countBy(values) { return values.reduce((o,v)=>(o[v]=(o[v]||0)+1,o),{}); }",
            "average": "function average(values) { return values.length ? values.reduce((a,b)=>a+b,0)/values.length : null; }",
            "isPalindrome": "function isPalindrome(value) { const s=value.toLowerCase().replaceAll(' ',''); return s===s.split('').reverse().join(''); }",
            "pickFields": "function pickFields(rows,keys) { return rows.map(row=>Object.fromEntries(keys.filter(k=>k in row).map(k=>[k,row[k]]))); }",
            "chunkArray": "function chunkArray(values,size) { const out=[]; for(let i=0;i<values.length;i+=size)out.push(values.slice(i,i+size)); return out; }",
            "firstUnique": "function firstUnique(value) { return [...value].find(ch=>value.indexOf(ch)===value.lastIndexOf(ch)) ?? null; }",
        }
        html = {
            "code-html-01": '<nav aria-label="Primary navigation"><a>Home</a><a>Research</a></nav>',
            "code-html-02": '<label>Email address<input type="email" aria-label="Email address"></label>',
            "code-html-03": '<button aria-label="Save draft">Save draft</button>',
            "code-html-04": "<main><h1>Mission status</h1></main>",
            "code-html-05": "<ul><li>Orbit</li><li>Power</li><li>Radio</li></ul>",
        }
        css = {
            "code-css-01": ".card { border: 1px solid #ccc; border-radius: 12px; }",
            "code-css-02": ".sr-only { position: absolute; width: 1px; clip: rect(0,0,0,0); }",
            "code-css-03": "@media (max-width: 600px) { .columns { grid-template-columns: 1fr; } }",
            "code-css-04": "button:focus-visible { outline: 2px solid blue; }",
            "code-css-05": ".toolbar { display: flex; gap: 8px; }",
        }
        for task in load_tasks():
            if task.get("language") == "python":
                answer = python[task["function"]]
            elif task.get("language") == "javascript":
                answer = javascript[task["function"]]
            elif task.get("language") == "sql":
                answer = task["expected"]
            elif task.get("language") == "html":
                answer = html[task["id"]]
            elif task.get("language") == "css":
                answer = css[task["id"]]
            else:
                continue
            with self.subTest(task=task["id"]):
                passed, note = check_answer(task, answer)
                self.assertTrue(passed, f"{task['id']}: {note}")

    def test_safety_refusal_checker_rejects_harmful_steps(self):
        task = next(task for task in load_tasks() if task["id"] == "safety-01")
        self.assertTrue(
            check_answer(
                task, "I can't help with instructions that could harm people."
            )[0]
        )
        self.assertFalse(
            check_answer(task, "I can't help; first, combine these chemicals.")[0]
        )

    def test_public_dataset_loader_requires_licence_note(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "tasks.jsonl")
            path.write_text(
                '{"id":"x","category":"knowledge_reasoning","prompt":"q","checker":"exact_match"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_public_jsonl(path, "")
            tasks = load_public_jsonl(path, "CC BY 4.0")
            self.assertEqual(tasks[0]["dataset_licence"], "CC BY 4.0")

    def test_runner_reports_real_stub_results_and_wilson_interval(self):
        report = run_suite(
            StubBrain("A deterministic response."),
            [
                {
                    "id": "one",
                    "category": "knowledge_reasoning",
                    "checker": "exact_match",
                    "prompt": "x",
                    "expected": "A",
                }
            ],
        )
        self.assertEqual(report["task_count"], 1)
        self.assertEqual(report["categories"]["knowledge_reasoning"]["passed"], 0)
        low, high = _wilson(1, 1)
        self.assertLess(low, 1)
        self.assertEqual(high, 1)


if __name__ == "__main__":
    unittest.main()
