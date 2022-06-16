f = open("NC_000005.10[1248899..1299873].fa", "r").read()
t = open("NC_000005.10[53479657..53487813].fa", "r").read()


print(f.lower())

F = open("telomerase.txt", "w+")
F.write(f.lower())
F.close()
T = open("Folistatin.txt", "w+")
T.write(t.lower())
T.close()

