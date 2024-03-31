
if __name__ == "__main__":
    with open("temp.txt") as f1:
        with open("6d300.txt") as f:
            for line in f:
                a, b, c, d, e, f = line.split()
                print(f"{a} {b} {c} {d} {e}")
