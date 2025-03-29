def is_prime(n):
    if n < 2:
        return False
    BASES_64BIT = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if n <= 40:
        return n in BASES_64BIT
    s, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        s += 1
    def check_base(a):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                return True
        return False
    return all(check_base(a) for a in BASES_64BIT)

def find_largest_3k_plus_2_prime(n):
    x = 6 * (n//6) + 5
    while x > 7:
        x -= 6
        if x % 5 == 0 or x % 7 == 0 or x % 11 == 0 or x % 13 == 0 or x % 17 == 0 or x % 19 == 0:
            continue
        if is_prime(x):
            return x
    return 0
