import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

TRJ = "runs/trajectory.npz"  # change if needed
data = np.load(TRJ)

S = data["states"]   # [T,4] -> x, x_dot, theta, theta_dot
A = data["actions"]  # [T]
Q0 = data["q_left"]  # [T]
Q1 = data["q_right"] # [T]

t = np.arange(len(S))

# --- time-series states
plt.figure()
plt.plot(t, S[:,0], label="cart x")
plt.plot(t, S[:,2], label="pole θ (rad)")
plt.xlabel("timestep"); plt.title("Cart & Pole States"); plt.legend(); plt.tight_layout()

# --- Q-values over time
plt.figure()
plt.plot(t, Q0, label="Q(left)")
plt.plot(t, Q1, label="Q(right)")
plt.xlabel("timestep"); plt.ylabel("Q"); plt.title("Q-Values Over Time"); plt.legend(); plt.tight_layout()

# --- simple animation (optional)
track_half = 2.4
cart_w, cart_h = 0.4, 0.2
pole_len = 1.0

fig, ax = plt.subplots()
ax.set_xlim(-2.7, 2.7); ax.set_ylim(-0.5, 1.2); ax.set_aspect('equal')
ax.set_title("CartPole Animation")
ax.plot([-track_half, track_half], [0,0], lw=2, alpha=0.5)

cart = plt.Rectangle((0,0), cart_w, cart_h, fill=False)
ax.add_patch(cart)
pole_line, = ax.plot([], [], lw=2)
text = ax.text(0.02, 0.92, '', transform=ax.transAxes)

def init():
    cart.set_xy((-cart_w/2, 0.0))
    pole_line.set_data([], [])
    text.set_text('')
    return cart, pole_line, text

def animate(i):
    x, theta = S[i,0], S[i,2]
    cart.set_xy((x - cart_w/2, 0.0))
    tip_x = x + pole_len * np.sin(theta)
    tip_y = cart_h/2 + pole_len * np.cos(theta)
    pole_line.set_data([x, tip_x], [cart_h/2, tip_y])
    text.set_text(f"t={i} θ={theta:+.3f} a={'L' if A[i]==0 else 'R'}")
    return cart, pole_line, text

ani = animation.FuncAnimation(fig, animate, frames=len(S), init_func=init, blit=True, interval=20)
plt.show()

# To export MP4 (requires ffmpeg):
# ani.save("runs/cartpole_demo.mp4", fps=50)
