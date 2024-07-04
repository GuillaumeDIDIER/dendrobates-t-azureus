import sys

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <input.csv> <mapping>")
    sys.exit(1)

input_file = sys.argv[1]
mapping_file = sys.argv[2]

mapping = []
with open(mapping_file, "r") as f:
    for i in f.read().split("\n"):
        if i != "":
            mapping.append(int(i))


with open(input_file, "r") as f:
    for line in f.read().split("\n"):
        if line == "" or "core" in line:
            print(line)
            continue

        sock, core, ht = map(int, line.split(","))
        core = mapping[core]
        print(f"{sock},{core},{ht}")
