import unittest

from backend.main import make_project_plan


class ProjectPlanTests(unittest.TestCase):
    def test_plan_has_slug_and_five_steps(self):
        plan = make_project_plan("Build a responsive portfolio website with a contact form")
        self.assertEqual(plan["slug"], "build-a-responsive-portfolio-website")
        self.assertEqual(len(plan["steps"]), 5)
        self.assertEqual(plan["status"], "draft_plan")

    def test_mobile_brief_gets_mobile_platform_guidance(self):
        plan = make_project_plan("Build an iOS mobile application for tracking habits")
        self.assertIn("mobile target", plan["steps"][0])

    def test_whitespace_is_normalized(self):
        plan = make_project_plan("  Build   a task manager  ")
        self.assertEqual(plan["summary"], "Build a task manager")


if __name__ == "__main__":
    unittest.main()
