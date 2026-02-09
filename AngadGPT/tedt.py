with open("std.txt", "w") as f:
    lines = [...]  # list of Q/A strings from above
    while len(lines) < 2000:
        lines += lines  # duplicate until enough
    for line in lines[:2000]:
        f.write(str(line))