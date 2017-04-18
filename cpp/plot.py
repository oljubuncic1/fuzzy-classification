import matplotlib.pyplot as plt
import sys
import ast

print("Jello from python")
arr = ast.literal_eval( sys.argv[1] )
cut = float(sys.argv[2])

x = [d[0] for d in arr]
y = [d[1] for d in arr]
colors = [d[2] for d in arr]

plt.scatter(x, y, c=colors)
if cut != -1:
    plt.axvline(cut)
plt.show()
