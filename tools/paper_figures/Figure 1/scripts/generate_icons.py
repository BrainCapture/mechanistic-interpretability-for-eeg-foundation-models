import matplotlib.pyplot as plt
import numpy as np

# Amplitude icon
fig, ax = plt.subplots(figsize=(1.5, 0.6))
bars = 5
x = np.arange(bars)
y = 1 / (x + 1)  # 1/f shape

ax.bar(x, y, color='#2D6FB5', width=0.8, align='center')
ax.set_ylim(0, 1.1)

# Keep x and y axis lines, but hide top and right
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#888')
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_color('#888')
ax.spines['left'].set_linewidth(1)

# Remove actual tick text/marks for a cleaner icon
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout(pad=0.1)
fig.savefig('amplitude.svg', transparent=True, bbox_inches='tight', pad_inches=0.01)
plt.close(fig)

# Phase icon
fig, ax = plt.subplots(figsize=(0.8, 0.8))
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), color='#bbb', linewidth=1)
ax.plot([-1.1, 1.1], [0, 0], color='#ddd', linewidth=1)
ax.plot([0, 0], [-1.1, 1.1], color='#ddd', linewidth=1)

# Draw an arrow for the phase (up right)
ang = np.pi / 4  # 45 degrees
ax.arrow(0, 0, np.cos(ang)*0.8, np.sin(ang)*0.8, head_width=0.15, head_length=0.15, fc='#2D6FB5', ec='#2D6FB5', linewidth=1.5)
ax.scatter([np.cos(ang)*0.95], [np.sin(ang)*0.95], color='#2D6FB5', s=15, zorder=5)

ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout(pad=0)
fig.savefig('phase.svg', transparent=True, bbox_inches='tight', pad_inches=0)
plt.close(fig)
