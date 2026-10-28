import unittest

from hardware_aware_selective_feedback.score_pelm import (
    TOKEN_CAP,
    answer_finished,
    clip_committed,
    few_shot_prompt,
    format_worked,
    scored_answer,
)


class AnswerStopTests(unittest.TestCase):
    def test_first_answer_survives_a_later_question(self):
        text = (
            "She has 9 eggs left. 9 * 2 = 18.\n"
            "The answer is 18\n"
            "Question: A train travels 80 miles in 2 hours.\n"
            "Answer: The speed is 40.\n"
        )
        self.assertEqual(scored_answer(text), "18")
        self.assertTrue(answer_finished(text))
        self.assertFalse(answer_finished("The answer is 18"))

    def test_missing_phrase_is_not_an_answer(self):
        text = "The speed of the train is 40 miles per hour.\n"
        self.assertEqual(scored_answer(text), "")
        self.assertFalse(answer_finished(text))

    def test_clip_stops_at_the_newline_after_the_number(self):
        pieces = {1: "The answer is ", 2: "230", 3: "\n", 4: "Question"}

        def decode(ids):
            return "".join(pieces[item] for item in ids)

        self.assertEqual(clip_committed(decode, [], [1, 2, 3, 4]), [1, 2, 3])
        self.assertEqual(clip_committed(decode, [], [1, 2]), [1, 2])

    def test_eight_train_examples_end_with_the_answer(self):
        demos = [(f"Question {index}?", f"Work.\n#### {index}") for index in range(8)]
        prompt = few_shot_prompt(demos, "How many apples?")
        self.assertEqual(prompt.count("The answer is"), 8)
        self.assertTrue(prompt.endswith("Question: How many apples?\nAnswer:"))
        self.assertNotIn("<<", format_worked("Q?", "Sees <<2+2=4>>4.\n#### 4"))
        self.assertIn("The answer is 4", format_worked("Q?", "Sees <<2+2=4>>4.\n#### 4"))
        self.assertEqual(TOKEN_CAP, 256)
        with self.assertRaises(ValueError):
            few_shot_prompt(demos[:7], "How many apples?")


if __name__ == "__main__":
    unittest.main()
