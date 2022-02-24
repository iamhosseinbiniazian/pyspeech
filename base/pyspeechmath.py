def Gcd(m,n): 
    if m==0 or n==0:
        if m==0 and n==0:
            raise TypeError("Undefined GCD since m = 0, n = 0.")
        return ((n if n > 0 else -n) if m == 0 else (m if m > 0 else -m))
    while (1):
        m %= n
        if m == 0:
            return (n if n > 0 else -n)
        n %= m
        if n == 0:
            return (m if m > 0 else -m)
#######################################################
def Lcm(m,n):
    if m<=0 or n<=0:
        raise "m and n most be greater than zero."
    gcd=Gcd(m,n)
    return gcd*(m/gcd)*(n/gcd)
###############################################