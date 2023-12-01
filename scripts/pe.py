import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from matplotlib.widgets import Slider

from enc import IPE, PE, AltEnc

mosaic = [['A', 'B', 'C'],
          ['A', 'B', 'C'],
          ['A', 'B', 'C'],
          ['a', 'b', 'b'],
          ['x_a', 'x_b_std', 'x_b_std'],
          ['L', 'L', 'L']]
fig, axs = plt.subplot_mosaic(mosaic, layout='constrained')

ax_pe, ax_pe_x = axs['A'], axs['a']
ax_ipe, ax_ipe_x = axs['B'], axs['b']
ax_alt = axs['C']

def make_slider(axis, label: str, val_name: str, val_min: float, val_max: float, val_init=0.0, val_step=0.01):
    slider = Slider(axis, label, val_min, val_max, valinit=val_init, valstep=val_step)

    def update(v):
        fig_state[val_name] = v
        draw(fig_state)

    slider.on_changed(update)
    return slider, update

x_min, x_max = -1, 1
L_max = 15

slider_L, update_L = make_slider(axs['L'], '$L$', 'L', 1, L_max, val_init=1, val_step=1)
slider_x, update_x = make_slider(axs['x_a'], '$x$', 'x', x_min, x_max, val_init=0, val_step=0.0001)
slider_sigma, update_sigma = make_slider(axs['x_b_std'], '$\sigma$', 'sigma', 0.0001, 1, val_init=0, val_step=0.01)

fig_state = {'x': 0, 'L': 1, 'sigma': 0.1}

def draw(state):
    encoding_axs = [ax_pe, ax_ipe, ax_alt]
    input_axs = [ax_pe_x, ax_ipe_x]

    # clear axes
    [ax.clear() for ax in encoding_axs + input_axs]

    # plot gaussian and delta function
    x_gaussian = np.linspace(x_min, x_max, 10000)
    y_gaussian = scipy.stats.norm.pdf(x_gaussian, loc=state['x'], scale=state['sigma'])
    y_delta = scipy.stats.norm.pdf(x_gaussian, loc=state['x'], scale=0.0001)
    ax_ipe_x.plot(x_gaussian, y_gaussian)
    ax_pe_x.plot(x_gaussian, y_delta)

    for ax in input_axs:
        ax.set_xlim(x_min, x_max)
        ax.axis('off')

    # compute PE and IPE
    pe = PE(1, state['L'])
    ipe = IPE(1, state['L'])
    alt = AltEnc(1, state['L'])

    x_pe = torch.Tensor([state['x']]).unsqueeze(0)
    y_pe = pe.encode(torch.Tensor(x_pe), torch.zeros(1, 1)).squeeze(0)

    mu_ipe = torch.Tensor([state['x']]).reshape(1, 1)
    cov_ipe = torch.Tensor([state['sigma']]).reshape(1, 1, 1)

    y_ipe = ipe.encode(mu_ipe, cov_ipe).squeeze()

    y_alt = alt.encode(mu_ipe, cov_ipe).squeeze()

    # plot PE and IPE
    bar_indices = range(len(y_pe))
    colors = [i // 2 for i in bar_indices]
    colors = [cm.viridis(plt.Normalize(0, L_max)(c)) for c in colors]

    ax_pe.bar(bar_indices, y_pe, color=colors)
    ax_ipe.bar(bar_indices, y_ipe, color=colors)

    # plot alt
    bar_indices_alt = range(len(y_alt))
    colors_alt = [i // 2 for i in bar_indices_alt]
    colors_alt = [cm.viridis(plt.Normalize(0, L_max)(c)) for c in colors_alt]

    ax_alt.bar(bar_indices_alt, y_alt, color=colors_alt)

    for ax in encoding_axs:
        ax.set_ylim(-1, 1)

    # clear ticks
    for ax in input_axs + encoding_axs:
        ax.set_xticks([])
        ax.set_yticks([])

draw(fig_state)
plt.tight_layout()
plt.show()
