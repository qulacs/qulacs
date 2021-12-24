import unittest
from tashizan import tashizan


class TestTashizan(unittest.TestCase):
    """test class of tashizan.py
    """

    def test_tashizan(self):
        """test method for tashizan
        """
        value1 = 2
        value2 = 6
        expected = 8
        actual = tashizan(value1, value2)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
