"""Answer extraction and normalisation (essaim/answers.py): regression cases from the audits."""
import unittest

from essaim.answers import extract, final_math, mc_letter, norm_math


class TestFinalMath(unittest.TestCase):
    CASES = [
        (r"The simplified expression is $15x - 80$.", "15x-80"),
        (r"The value of $n$ is 8.", "8"),
        (r"Thus $x = 7$.", "7"),
        (r"Thus $x = 7$ is correct.", "7"),
        (r"The answer is 12.", "12"),
        ("Therefore,\n$$ \\sqrt{2} $$", r"\sqrt{2}"),
        (r"We get $x_1 = \frac{3}{4}$", r"\frac{3}{4}"),
        (r"Hence $\frac{3}{4}$ is the probability.", r"\frac{3}{4}"),
        (r"The result is $$2+3$$, so the answer is $$5$$.", "5"),
        (r"x = \sqrt{2}", r"\sqrt{2}"),
        (r"So the answer: $(2, -1)$", "(2,-1)"),
        ("We are asked to find $b+c$:\n$$b+c = 1.3 + 1.91$$\n$$b+c = 3.21$$", "3.21"),
        (r"The value of $p(0) + p(4)$ is 24, regardless of the value of $r$.", "24"),
        (r"Thus, $\dbinom{31}{28} = 4495$.", "4495"),
        (r"$$K = \frac{1}{2} \|v\| = \frac{1}{2} (6\sqrt{5}) = 3\sqrt{5}$$", r"3\sqrt{5}"),
        (r"$$\text{Constant Term} = -125$$", "-125"),
        ("Equating the exponents:\n$$-n = -8$$\n$$n = 8$$", "8"),
        (r"Since we chose the smallest $m$, 27 is the smallest such number.", "27"),
    ]

    def test_cases(self):
        for text, want in self.CASES:
            with self.subTest(text=text):
                self.assertEqual(norm_math(final_math(text)), norm_math(want))

    def test_equations_and_symbols(self):
        self.assertEqual(norm_math(final_math(r"The plane is $5x - 7y + 11z = 4$.", equation_asked=True)),
                         norm_math("5x-7y+11z=4"))
        self.assertEqual(norm_math(final_math(r"So the plane is $5x - 7y + 11z + 4 = 0$.", equation_asked=True)),
                         norm_math("5x-7y+11z+4=0"))
        self.assertEqual(final_math(r"The answer is $i$."), "i")
        self.assertEqual(final_math(r"The answer is $x_1$."), "x_1")
        q = {"question": "Find the equation of the plane through ..."}
        self.assertEqual(extract("math500", r"The plane is $2x + y = 3$.", q), norm_math("2x+y=3"))
        q = {"question": r"Enter your answer in the form $Ax + By + Cz + D = 0$."}
        self.assertEqual(extract("math500", r"So $x - 2y + z + 1 = 0$.", q), norm_math("x-2y+z+1=0"))
        mentioned = {"question": r"The equation of the line is $y = mx + b$. Find $a + b + m$."}
        self.assertEqual(extract("math500", r"Therefore $a + b + m = 13$.", mentioned), "13")

    def test_boxed_wins(self):
        self.assertEqual(extract("math500", r"so $x=3$, \boxed{\dfrac{1}{2}}", {}), r"\frac{1}{2}")


class TestNormMath(unittest.TestCase):
    def test_units_and_numbers(self):
        self.assertEqual(norm_math(r"5.4 \text{ cents}"), "5.4")
        self.assertEqual(norm_math(r"\frac{1}{2}\text{ meters}"), r"\frac{1}{2}")
        self.assertEqual(norm_math(r"\text{ Evelyn}"), "Evelyn")
        self.assertEqual(norm_math("0.50"), r"\frac{1}{2}")
        self.assertEqual(norm_math("1,000"), "1000")


class TestOtherBenches(unittest.TestCase):
    def test_gsm8k_last_statement(self):
        self.assertEqual(extract("gsm8k", "The answer is 4.\nCorrection: The answer is **16.**", {}), "16")
        self.assertEqual(extract("gsm8k", "The answer is $N = 80$.", {}), "80")

    def test_mc_letter(self):
        self.assertIsNone(mc_letter("I do not know.", list("ABCDEFGHIJ")))
        self.assertEqual(mc_letter("The answer is (C).", list("ABCD")), "C")
        self.assertEqual(mc_letter("**B**", list("ABCD")), "B")


if __name__ == "__main__":
    unittest.main()
