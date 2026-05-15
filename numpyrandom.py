import numpy as np;

def main():
    a = np.random.randn(1,3)
    # print(a)

    b = np.random.randn(3, 3)
    # print(b)

    c = a * b
    print(c.shape)

    d = a + b.transpose()
    # print(d)

    k=np.array([[2,1],[1,3]])
    print(k*k)


if __name__ == '__main__':
    main()