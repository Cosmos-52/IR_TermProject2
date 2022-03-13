import main
from textblob import TextBlob


def test_case_one(data):
    result = main.query_by_title(data)
    if testdata1 in result:
        print("test case one pass")
    else:
        print("test case one fail")


def test_case_two(data, type):
    result = main.check_spell(data,type)
    if testdata1 in result:
        print("test case two pass")
    else:
        print("test case two fail")


def test_case_three(wrongspell, correctspell):
    check = TextBlob(wrongspell).correct()
    correct = str(check)
    if correct == correctspell:
        print("test case three pass")
    else:
        print("test case three fail")


if __name__ == '__main__':
    testdata1 = 'mushr'
    test_case_one(testdata1)

    testdata2 = 'salt'
    test_case_two(testdata2, 'ingredients')

    testdata3_1 = 'mushro'
    testdata3_2 = 'mushroom'
    test_case_three(testdata3_1, testdata3_2)