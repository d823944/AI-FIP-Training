def myFactorial(N):
    """calculate factorial
    Formula: N! = 1 * 2 * ... * N
    """
    if N == 1 or N == 0:
        return 1
    else:
        return N * myFactorial(N - 1)


def myCombinations(n, r):
    """calculate combinations
    Formula: nCr = n! / r! * (n - r)!
    """
    if (n is None) or (r is None) or (n < r):
        return None
    else:
        result = myFactorial(n) / myFactorial(r) / myFactorial(n-r)
        return int(result)

if __name__ == '__main__':
    print(myFactorial(6))
    print(myCombinations(6, 4))