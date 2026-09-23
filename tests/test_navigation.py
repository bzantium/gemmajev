"""Check local observation boundaries and necessary backtracking."""

import unittest

from gemmajev.navigation import NavigationMemory, select_action


class NavigationTest(unittest.TestCase):
    def test_only_observed_open_unvisited_directions_are_available(self):
        memory = NavigationMemory((2, 2), (4, 4), 5)
        memory.observe("Local map:\n#####\n##.##\n#.A.#\n#####\n#####")
        self.assertEqual(memory.available(), ["north", "east", "west"])
        chosen = select_action(dict(north=0.1, east=0.2, south=0.6, west=0.1), memory.available())
        self.assertEqual(chosen, "east")

    def test_dead_end_backtracks_instead_of_forbidding_all_revisits(self):
        memory = NavigationMemory((2, 2), (4, 4), 5)
        memory.observe("Local map:\n#####\n#####\n##A.#\n#####\n#####")
        memory.transition("east", (2, 3), False)
        memory.observe("Local map:\n####X\n####X\n#.A#X\n####X\n####X")
        self.assertEqual(memory.available(), ["west"])
        request = memory.request("dead-end")
        self.assertIn("west: open; visits=1", request["state"])
        memory.transition("west", (2, 2), False)
        self.assertEqual(memory.stack, [(2, 2)])
        self.assertEqual(memory.visits[(2, 2)], 2)

    def test_memory_changes_request_at_the_same_coordinate(self):
        memory = NavigationMemory((2, 2), (4, 4), 5)
        local = "Local map:\n#####\n#####\n##A.#\n#####\n#####"
        memory.observe(local)
        before = memory.request("same-position")["state"]
        memory.transition("east", (2, 3), False)
        memory.transition("west", (2, 2), False)
        memory.observe(local)
        self.assertNotEqual(before, memory.request("same-position")["state"])

    def test_actual_environment_rollout_can_be_recorded_and_replayed(self):
        from examples.navigate import run_case

        class Engine:
            def predict(self, request, batch_questions):
                return {
                    "states": [
                        {
                            "answers": {
                                "move": {
                                    "probabilities": dict(north=0.1, east=0.7, south=0.1, west=0.1)
                                }
                            }
                        }
                    ]
                }

        case = dict(
            id="tiny",
            scope="generated",
            initial_state=dict(game="scaled_maze", size=3, walls=[], position=[1, 1], goal=[1, 2]),
        )
        episode = run_case(case, Engine())
        self.assertTrue(episode["summary"]["goal"])
        self.assertTrue(episode["summary"]["replay_verified"])
        self.assertEqual(episode["summary"]["model_calls"], 1)
        self.assertEqual(episode["steps"][0]["position"], [1, 1])


if __name__ == "__main__":
    unittest.main()
