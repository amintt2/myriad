"""E12 part 3: the agent-loop skeleton (fake chat, fake environment):  uv run python -m unittest tests.test_agent"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from essaim import agent  # noqa: E402


def block(cmd: str, thought: str = "Thinking.") -> str:
    return f"{thought}\n```bash\n{cmd}\n```"


class FakeEnv:
    def __init__(self):
        self.ran = []

    def run(self, cmd, timeout_s):
        self.ran.append(cmd)
        return 0, f"ran {cmd}"

    def close(self):
        pass


class Parsing(unittest.TestCase):
    def test_exactly_one_block(self):
        self.assertEqual(agent.parse_action(block("ls -la")), "ls -la")
        for bad in ("no block", block("a") + block("b"), "```bash\n\n```"):
            with self.assertRaises(agent.ActionError):
                agent.parse_action(bad)

    def test_submit(self):
        self.assertTrue(agent.is_submit(f"echo {agent.SUBMIT}"))
        self.assertFalse(agent.is_submit(f"echo {agent.SUBMIT} && rm -rf /"))
        self.assertFalse(agent.is_submit(f"echo {agent.SUBMIT}\nrm -rf /"))
        self.assertFalse(agent.is_submit(""))

    def test_normalise(self):
        self.assertEqual(agent.normalise("ls   -la  # list"), agent.normalise("ls -la"))
        # quoting changes what the shell does: never merged
        self.assertNotEqual(agent.normalise("echo \"$HOME\""), agent.normalise("echo '$HOME'"))
        self.assertNotEqual(agent.normalise("ls *.py"), agent.normalise("ls '*.py'"))
        # several lines (heredoc): indentation is content
        self.assertNotEqual(agent.normalise("python - <<E\n  x = 1\nE"), agent.normalise("python - <<E\nx = 1\nE"))
        self.assertEqual(agent.normalise("\ncat f  \n"), agent.normalise("cat f"))
        self.assertNotEqual(agent.normalise("ls -la"), agent.normalise("ls -l"))
        self.assertTrue(agent.normalise("echo 'unclosed"))  # does not raise


class Voting(unittest.TestCase):
    def test_plurality_and_tie_priority(self):
        cmd, note = agent.vote([("a", "ls  -la"), ("b", "ls -la"), ("c", "pwd")], ["c", "b", "a"])
        self.assertEqual((cmd, note["chosen_votes"], note["unanimous"]), ("ls -la", 2, False))
        cmd, _ = agent.vote([("a", "ls"), ("b", "pwd")], ["b", "a"])
        self.assertEqual(cmd, "pwd")  # a tie goes to the family ranked first
        with self.assertRaises(agent.ActionError):
            agent.vote([], [])


class Episode(unittest.TestCase):
    def test_single_runs_until_submit(self):
        replies = iter([block("ls"), block("cat f"), block(f"echo {agent.SUBMIT}")])
        env, seen = FakeEnv(), []

        def chat(model, messages):
            seen.append((model, len(messages)))
            return next(replies)
        out = agent.run_episode(chat, env, "do it", ["m1"], "single", max_steps=10)
        self.assertEqual((out["stopped"], env.ran), ("submitted", ["ls", "cat f"]))
        self.assertEqual([m for m, _ in seen], ["m1"] * 3)
        self.assertEqual([n for _, n in seen], [2, 4, 6])  # the observation of each command is fed back

    def test_format_error_costs_a_turn_and_max_steps_stops(self):
        env = FakeEnv()
        out = agent.run_episode(lambda m, msgs: "I refuse to use a block", env, "t", ["m"], max_steps=3)
        self.assertEqual((out["stopped"], len(out["steps"]), env.ran), ("max_steps", 3, []))
        self.assertTrue(out["steps"][0].note["format_error"])

    def test_vote_asks_every_family_and_runs_the_winner(self):
        env = FakeEnv()
        said = {"f1": block("ls"), "f2": block("ls  "), "f3": block("rm x")}
        steps = [dict(said), {k: block(f"echo {agent.SUBMIT}") for k in said}]

        asked = []

        def chat(model, messages):
            asked.append(model)
            return steps[0 if len(asked) <= 3 else 1][model]
        out = agent.run_episode(chat, env, "t", ["f1", "f2", "f3"], "vote", max_steps=5)
        self.assertEqual(env.ran, ["ls"])
        self.assertEqual(out["steps"][0].note["chosen_votes"], 2)
        self.assertEqual(asked[:3], ["f1", "f2", "f3"])

    def test_unknown_strategy_and_output_clipping(self):
        with self.assertRaises(ValueError):
            agent.run_episode(lambda m, x: "", FakeEnv(), "t", ["m"], "verify")
        long = "x" * 10000
        self.assertLess(len(agent.clip(long)), 4100)
        self.assertEqual(agent.clip("short"), "short")


if __name__ == "__main__":
    unittest.main()
