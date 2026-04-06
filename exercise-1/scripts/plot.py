import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

p = [i / 1000 for i in range(1001)]

lines = [
    (p, [6 - 6*pi for pi in p], r"$u_2(\mathbf{p},\, e^1) = 6 - 6p$"),
    (p, [5 - 3*pi for pi in p], r"$u_2(\mathbf{p},\, e^2) = 5 - 3p$"),
    (p, [3 + pi for pi in p],   r"$u_2(\mathbf{p},\, e^3) = 3 + p$"),
    (p, [5*pi for pi in p],     r"$u_2(\mathbf{p},\, e^4) = 5p$"),
]

fig, ax = plt.subplots(figsize=(8, 5))

for xs, ys, label in lines:
    ax.plot(xs, ys, label=label, linewidth=2)

ax.set_xlabel("$p$")
ax.set_ylabel("Expected utility of player 2")
ax.set_xlim(0, 1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=11)
ax.set_title("Expected utility of player 2 vs. player 1's mixed strategy $\\mathbf{p} = (p,\\, 1-p)$")

plt.tight_layout()
plt.savefig("plot.pdf")
plt.show()
