from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.medha_data_factory.core import (
    deduplicate_jsonl, detect_frameworks, ingest_local_repository, inspect_repository,
    load_source_manifest, make_evaluation_record, make_preference_record,
    promote_reviewed_candidate, stable_split, validate_source_policy,
)
from backend.medha_data_factory.github import discover_repositories, download_selected_repository
from backend.medha_data_factory.swegym_adapter import adapt_swegym_row
from backend.medha_data_factory.scaling import build_progress_report, load_target_plan
from backend.medha_data_factory.quality import score_sft_candidate
from backend.medha_data_factory.shards import build_shards


class MedhaDataFactoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "demo-repo"
        (self.repo / "src").mkdir(parents=True)
        (self.repo / "tests").mkdir()
        (self.repo / "node_modules" / "package").mkdir(parents=True)
        (self.repo / "README.md").write_text("# Demo\nA small example repository for Medha factory testing.\n", encoding="utf-8")
        self.code = "def add(left: int, right: int) -> int:\n    return left + right\n"
        (self.repo / "src" / "main.py").write_text(self.code, encoding="utf-8")
        (self.repo / "src" / "duplicate.py").write_text(self.code, encoding="utf-8")
        (self.repo / "tests" / "test_main.py").write_text(
            "def test_add():\n    assert 1 + 1 == 2\n\n# A documented test file for ingestion.\n",
            encoding="utf-8",
        )
        (self.repo / "src" / "leaked.py").write_text(
            "TOKEN = 'ghp_" + "A" * 36 + "'\nprint('must not ingest')\n",
            encoding="utf-8",
        )
        (self.repo / ".env").write_text("API_TOKEN=should-not-ingest\n", encoding="utf-8")
        (self.repo / "node_modules" / "package" / "index.js").write_text("module.exports = 1;\n", encoding="utf-8")
        self.source = {
            "source_id": "example-org/demo-repo",
            "repository_url": "https://github.com/example-org/demo-repo",
            "revision": "a" * 40,
            "declared_license": "Apache-2.0",
            "underlying_license": "Apache-2.0",
            "usage_status": "approved_with_attribution",
            "license_evidence_url": "https://github.com/example-org/demo-repo/blob/main/LICENSE",
            "reviewed_at": "2026-11-16",
            "reviewer": "human-reviewer",
            "license_scope": "repository-reviewed",
            "attribution_text": "Example project under Apache-2.0.",
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_source_policy_blocks_unreviewed_and_unknown(self):
        with self.assertRaises(PermissionError):
            validate_source_policy(dict(self.source, usage_status="hold"))
        with self.assertRaises(PermissionError):
            validate_source_policy(dict(self.source, underlying_license="unknown"))

    def test_manifest_requires_explicit_source_and_approval(self):
        manifest = self.root / "sources.json"
        manifest.write_text(json.dumps({"manifest_version": "1.0.0", "sources": [self.source]}), encoding="utf-8")
        self.assertEqual(load_source_manifest(manifest, self.source["source_id"])["repository_url"], self.source["repository_url"])
        with self.assertRaises(KeyError):
            load_source_manifest(manifest, "not-listed")
        held = dict(self.source, usage_status="hold")
        manifest.write_text(json.dumps({"manifest_version": "1.0.0", "sources": [held]}), encoding="utf-8")
        with self.assertRaises(PermissionError):
            load_source_manifest(manifest, self.source["source_id"])

    def test_framework_detection_reads_dependency_manifests(self):
        (self.repo / "package.json").write_text(
            json.dumps({"dependencies": {"react": "^19.0.0", "next": "^15.0.0", "react-native": "^0.8"}}),
            encoding="utf-8",
        )
        (self.repo / "requirements.txt").write_text("fastapi>=0.110\\npytest>=8.0\\n", encoding="utf-8")
        frameworks = detect_frameworks(self.repo)
        self.assertIn("react", frameworks)
        self.assertIn("nextjs", frameworks)
        self.assertIn("react-native", frameworks)
        self.assertIn("fastapi", frameworks)
        self.assertIn("pytest", frameworks)

    def test_deterministic_repository_split(self):
        self.assertEqual(stable_split("https://github.com/example/repo", 21), stable_split("https://github.com/example/repo", 21))
        self.assertIn(stable_split("https://github.com/example/repo", 21), {"train", "validation", "test"})

    def test_inspect_is_read_only(self):
        report = inspect_repository(self.repo)
        self.assertGreaterEqual(report["file_count"], 3)
        self.assertFalse(report["execution_performed"])
        self.assertFalse(report["network_access_performed"])
        self.assertIn("python", report["languages"])

    def test_ingestion_filters_deduplicates_and_validates_schema(self):
        out = self.root / "out"
        report = ingest_local_repository(self.repo, self.source, out, seed=17)
        self.assertGreaterEqual(report["records_written"], 3)
        self.assertEqual(report["duplicates_removed"], 1)
        self.assertFalse(report["execution_performed"])
        self.assertFalse(report["network_access_performed"])
        records = [json.loads(line) for line in (out / f"pretraining_{report['dataset_split']}.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(records)
        for record in records:
            self.assertEqual(record["dataset_split"], report["dataset_split"])
            self.assertEqual(record["provenance"]["usage_status"], "approved_with_attribution")
            self.assertEqual(len(record["provenance"]["content_sha256"]), 64)
        all_paths = {r["source_metadata"]["relative_path"] for r in records}
        self.assertNotIn("src/leaked.py", all_paths)
        self.assertNotIn(".env", all_paths)
        self.assertNotIn("node_modules/package/index.js", all_paths)
        self.assertIn("secret_pattern_detected", (out / "rejections.jsonl").read_text(encoding="utf-8"))
        candidates = [json.loads(line) for line in (out / "task_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertGreaterEqual(len(candidates), 2)
        self.assertTrue(all(x["training_eligible"] is False for x in candidates))

    def test_output_directory_must_be_empty(self):
        output = self.root / "not-empty"
        output.mkdir()
        (output / "keep.txt").write_text("do not overwrite", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            ingest_local_repository(self.repo, self.source, output)

    def test_sft_promotion_requires_verification_and_human_review(self):
        index = {
            "repository_id": self.source["repository_url"],
            "repository_url": self.source["repository_url"],
            "revision": "a" * 40,
            "declared_license": self.source["declared_license"],
            "underlying_license": self.source["underlying_license"],
            "usage_status": self.source["usage_status"],
            "license_scope": self.source["license_scope"],
            "license_evidence_url": self.source["license_evidence_url"],
            "reviewer": self.source["reviewer"],
            "reviewed_at": self.source["reviewed_at"],
            "dataset_split": "train",
            "seed": 42,
        }
        candidate = {"candidate_id": "example:debug", "task_type": "debugging", "prompt": "Fix the documented bug."}
        review = {
            "assistant_answer": "Use a guarded access and add regression tests.",
            "human_reviewed": True, "safety_reviewed": True, "factuality_pass": True,
            "requires_execution": True, "tests_passed": 2, "build_passed": True,
            "verification_report": {
                "status": "passed", "exit_code": 0, "network_enabled": False,
                "source_mount_read_only": True, "repository_code_executed": True,
                "image": "python:3.12-slim", "command": "python -m unittest",
            },
            "reviewer_id": "reviewer-1", "reviewed_at": "2026-11-16",
        }
        record = promote_reviewed_candidate(candidate, review, index)
        self.assertEqual(record["task_type"], "debugging")
        self.assertGreaterEqual(record["quality_checks"]["quality_score"], 85)
        self.assertTrue(record["quality_checks"]["training_eligible"])
        self.assertTrue(record["quality_checks"]["human_reviewed"])
        with self.assertRaises(PermissionError):
            promote_reviewed_candidate(candidate, dict(review, tests_passed=0), index)
        with self.assertRaises(PermissionError):
            promote_reviewed_candidate(candidate, review, dict(index, dataset_split="test"))
        with self.assertRaises(PermissionError):
            promote_reviewed_candidate(candidate, dict(review, safety_reviewed=False), index)

    def test_preference_pair_uses_direction_and_rejects_tie(self):
        source = dict(self.source, source_split="train")
        record = make_preference_record(
            record_id="pref-1", prompt="Fix this Python bug.",
            response_1="Incorrect fix.", response_2="Correct fix with a test.",
            preference_strength=2, preference_reason="Response 2 is more correct.",
            source=source, domain_tags=["software_engineering", "debugging"],
        )
        self.assertEqual(record["chosen"], "Correct fix with a test.")
        self.assertEqual(record["rejected"], "Incorrect fix.")
        with self.assertRaises(ValueError):
            make_preference_record(
                record_id="pref-tie", prompt="P", response_1="A", response_2="B",
                preference_strength=0, preference_reason="Tie.", source=source,
            )

    def test_evaluation_records_are_test_only(self):
        source = dict(self.source, dataset_split="test")
        record = make_evaluation_record(
            task_id="eval-1", category="coding", prompt="Implement a safe parser.",
            reference_answer="A verified implementation.", rubric=[
                {"criterion":"Correctness","weight":0.7,"description":"Passes tests."},
                {"criterion":"Security","weight":0.3,"description":"Rejects malformed input."},
            ], source=source, hidden_tests=["test_valid", "test_invalid"],
        )
        self.assertEqual(record["split"], "test")
        self.assertTrue(record["source_metadata"]["sealed"])
        with self.assertRaises(PermissionError):
            make_evaluation_record(
                task_id="eval-2", category="coding", prompt="P", reference_answer="A",
                rubric=[{"criterion":"correct","weight":1,"description":"works"}],
                source=dict(self.source, dataset_split="train"),
            )

    def test_dedupe_prefers_test_when_same_content_crosses_splits(self):
        train = {"record_id":"train-1","text":"same code","dataset_split":"train","provenance":{"content_sha256":"b"*64}}
        test = {"record_id":"test-1","text":"same code","dataset_split":"test","provenance":{"content_sha256":"b"*64}}
        train_path, test_path = self.root/"train.jsonl", self.root/"test.jsonl"
        train_path.write_text(json.dumps(train)+"\n", encoding="utf-8")
        test_path.write_text(json.dumps(test)+"\n", encoding="utf-8")
        out, ledger = self.root/"unique.jsonl", self.root/"dupes.jsonl"
        report = deduplicate_jsonl([train_path,test_path],out,ledger)
        kept = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(kept["record_id"], "test-1")
        self.assertEqual(report["duplicates_removed"], 1)

    def test_swegym_adapter_requires_both_licence_clearances(self):
        dataset_source = dict(self.source, source_id="SWE-Gym", repository_url="https://huggingface.co/datasets/SWE-Gym/SWE-Gym")
        row = {
            "instance_id":"task-1", "problem_statement":"Handle empty input safely.",
            "patch":"diff --git a/app.py b/app.py", "test_patch":"diff --git a/test.py b/test.py",
            "repo":"example-org/demo-repo", "base_commit":"c"*40,
            "FAIL_TO_PASS":["tests/test_empty.py"], "PASS_TO_PASS":["tests/test_existing.py"],
            "split":"train",
        }
        candidate = adapt_swegym_row(row, dataset_source, self.source)
        self.assertFalse(candidate["training_eligible"])
        self.assertFalse(candidate["verification"]["fail_to_pass_passed"])
        with self.assertRaises(PermissionError):
            adapt_swegym_row(row, dataset_source, dict(self.source, usage_status="hold"))

    @patch("backend.medha_data_factory.github._request_json")
    def test_discovery_is_metadata_only(self, request_json):
        request_json.return_value = {"items":[{
            "full_name":"example-org/demo-repo","html_url":self.source["repository_url"],
            "description":"A sample project","stargazers_count":100,
            "forks_count":10,"language":"Python","license":{"spdx_id":"Apache-2.0"},
            "archived":False,"size":123,"default_branch":"main","pushed_at":"2026-11-01T00:00:00Z",
        }]}
        results = discover_repositories("topic:fastapi", limit=5, allowed_licenses={"Apache-2.0"})
        self.assertEqual(len(results),1)
        self.assertTrue(results[0]["metadata_only"])
        self.assertTrue(results[0]["requires_human_license_review"])

    @patch("backend.medha_data_factory.github._request_bytes")
    @patch("backend.medha_data_factory.github._request_json")
    def test_harvest_uses_pinned_commit_and_skips_secrets(self, request_json, request_bytes):
        safe = b"def hello():\n    return 'hello medha'\n"
        request_json.side_effect = [
            {"private":False,"archived":False,"disabled":False,"license":{"spdx_id":"Apache-2.0"},"default_branch":"main","html_url":self.source["repository_url"]},
            {"object":{"type":"commit","sha":"c"*40}},
            {"tree":{"sha":"d"*40}},
            {"truncated":False,"tree":[
                {"type":"blob","mode":"100644","path":"src/main.py","size":len(safe),"sha":"1"*40},
                {"type":"blob","mode":"100644","path":".env","size":20,"sha":"2"*40},
            ]},
        ]
        request_bytes.return_value=safe
        destination=self.root/"harvest"
        harvest_source = dict(self.source, revision=None)
        result=download_selected_repository("example-org/demo-repo",harvest_source,destination,max_files=10,max_file_bytes=1024,max_total_bytes=4096)
        self.assertEqual(result["revision"],"c"*40)
        self.assertEqual(result["files_downloaded"],1)
        self.assertIn("/"+("c"*40)+"/",request_bytes.call_args.args[0])
        self.assertTrue((destination/"src"/"main.py").exists())
        self.assertFalse((destination/".env").exists())

    def test_limits_are_enforced(self):
        with self.assertRaises(ValueError):
            ingest_local_repository(self.repo,self.source,self.root/"bad",max_files=5001)


    def test_100m_target_plan_and_progress_quotas(self):
        root = Path(__file__).resolve().parents[1]
        plan = load_target_plan(root / "datasets" / "medha_dataset_starter_pack" / "config" / "software_engineering_100m_plan.json")
        self.assertEqual(plan["target_unique_examples"], 100_000_000)
        self.assertEqual(sum(plan["language_primary_mix_percent"].values()), 100)
        self.assertEqual(sum(plan["task_type_mix_percent"].values()), 100)
        counts = {
            "unique_examples": 1000,
            "language_primary": {"python":200,"javascript":150,"typescript":150,"java":120,"cpp":120,"swift":80,"multi_language":180},
            "task_types": {
                "implementation_and_feature_delivery":180,"debugging_and_root_cause":160,
                "testing_and_validation":120,"repository_comprehension":100,
                "refactoring_and_maintainability":80,"backend_api_and_database":80,
                "frontend_mobile_and_accessibility":80,"architecture_and_system_design":70,
                "performance_security_and_reliability":70,"devops_delivery_and_observability":40,
                "documentation_and_maintenance":20
            },
            "full_stack_examples":200,
            "splits":{"train":900,"validation":50,"test":50},
            "quality_passed_examples":950,
            "licence_approved_examples":1000
        }
        report = build_progress_report(plan, counts)
        self.assertEqual(report["unique_examples"], 1000)
        self.assertEqual(report["remaining_examples"], 99_999_000)
        self.assertEqual(report["language_primary_quotas"]["python"]["target"], 20_000_000)
        self.assertEqual(report["full_stack"]["minimum_target"], 20_000_000)
        self.assertEqual(report["full_stack"]["required_at_current_scale"], 200)
        self.assertEqual(report["status"], "in_progress")
        self.assertIn("Some counted examples have not passed the quality gate.", report["warnings"])

    def test_quality_scorer_rejects_unreviewed_candidate(self):
        candidate = {"candidate_id":"candidate-1","prompt":"Implement a safe parser and explain its edge cases."}
        review = {
            "assistant_answer":"A reviewed reference implementation with validation and useful error handling.",
            "human_reviewed":False,"factuality_pass":True,"safety_reviewed":True,
            "requires_execution":True,"build_passed":True,"tests_passed":2,
            "verification_report":{"status":"passed","exit_code":0,"network_enabled":False,
                "source_mount_read_only":True,"repository_code_executed":True}
        }
        index = dict(self.source, revision="a"*40)
        score = score_sft_candidate(candidate, review, index)
        self.assertFalse(score["eligible"])
        self.assertIn("human_reviewed must be true", score["blockers"])



    def test_streaming_shards_remove_duplicates_across_splits(self):
        source_train = self.root / "input-train.jsonl"
        source_test = self.root / "input-test.jsonl"
        train_rows = [
            {"record_id":"train-a","text":"same content","dataset_split":"train"},
            {"record_id":"train-b","text":"two","dataset_split":"train"},
            {"record_id":"train-c","text":"three","dataset_split":"train"}
        ]
        test_rows = [
            {"record_id":"test-a","text":"same content","dataset_split":"test"},
            {"record_id":"test-b","text":"four","dataset_split":"test"}
        ]
        source_train.write_text("".join(json.dumps(row)+"\n" for row in train_rows),encoding="utf-8")
        source_test.write_text("".join(json.dumps(row)+"\n" for row in test_rows),encoding="utf-8")
        output = self.root / "shards"
        manifest = build_shards([source_train, source_test], output, examples_per_shard=1)
        self.assertEqual(manifest["input_rows"], 5)
        self.assertEqual(manifest["examples_written"], 4)
        self.assertEqual(manifest["duplicate_rows_removed"], 1)
        self.assertEqual(manifest["cross_split_duplicate_fingerprints"], 1)
        self.assertEqual(manifest["split_counts"], {"train": 2, "validation": 0, "test": 2})
        train_content = "".join(p.read_text(encoding="utf-8") for p in sorted(output.glob("train-*.jsonl")))
        test_content = "".join(p.read_text(encoding="utf-8") for p in sorted(output.glob("test-*.jsonl")))
        self.assertNotIn("train-a", train_content)
        self.assertIn("test-a", test_content)
        self.assertEqual(len(list(output.glob("train-*.jsonl"))), 2)
        self.assertEqual(len(list(output.glob("test-*.jsonl"))), 2)
        self.assertTrue((output / "manifest.json").exists())



if __name__ == "__main__":
    unittest.main()
