def myAdder(*args):
    """ calculate continued addition of multiple numbers
    """
    if len(args) < 1:
        return None
    else:
        result = 0
        for num in args:
            result += num
        return result

def myMult(*args):
    """ calculate continued multiplication of multiple numbers
    """
    if len(args) < 1:
        return None
    else:
        result = 1
        for num in args:
            result *= num
        return result
    

if __name__ == '__main__':
    print(myAdder(1, 2, 3, 4, 5))
    print(myMult(1, 3, 5, 7))