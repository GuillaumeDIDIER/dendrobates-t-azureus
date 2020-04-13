import pandas

columns = ["Addr", "Hash"]
core_number = 8 # FIXME
for i in range(0, core_number):
    for stat in ["Min", "Med", "Max"]:
        for op in ["Hit", "Miss"]:
            columns.append(op + str(i) + stat)
columns.append("Hmm")
df = pandas.read_csv("citron-vert/combined.csv", header=0, names=columns)
selected_columns = columns[:-1]
df = df[selected_columns]
print(df.head())

