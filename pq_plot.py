import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scienceplots

# Import netCDF file
df = pd.read_parquet("./test.parquet")

# Prepare Data to Plot
x = df['x']
y = df['y']
y_hat = df['y_test']
y_up = df['y_up']
y_down = df['y_down']
l1 = df['l1']
l1_signed = df['l1_s']
psqi = df['psqi']
psqi_t = df['psqi_t']
psqi_s = df['psqi_s']
psqi_st = df['psqi_st']
ub = np.array(df['ub'])
ubt = np.array(df['ub_t'])
ubs = np.array(df['ub_s'])
ubst = np.array(df['ub_st'])

ub_vec = np.array([ub, ubt, ubs, ubst])

# Plot params
pparam = dict(
    xlabel = r'$x$',
    ylabel = r'$y$',
    xscale = 'linear',
    yscale = 'linear',
    ylim = (0, ub_vec.max() + 0.1),
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(x, l1, label=r'$L_1$')
    ax.plot(x, ub, 'r-', label=r'Upper Bound')
    ax.plot(x, ubt, 'g--', label=r'Upper Bound Trimmed')
    ax.plot(x, ubs, 'b-.', label=r'Upper Bound Signed')
    ax.plot(x, ubst, 'k:', label=r'Upper Bound Signed Trimmed')
    ax.legend()
    fig.savefig('l1_plot.png', dpi=300, bbox_inches='tight')

# Plot params
pparam = dict(
    xlabel = r'$x$',
    ylabel = r'$y$',
    xscale = 'linear',
    yscale = 'linear',
    ylim = (-ub_vec.max() - 0.1, ub_vec.max() + 0.1),
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(x, l1_signed, label=r'$L_1$ Signed')
    ax.plot(x, ub, 'r-', label=r'Upper Bound')
    ax.plot(x, -ub, 'r-')
    ax.plot(x, ubt, 'g--', label=r'Upper Bound Trimmed')
    ax.plot(x, -ubt, 'g--')
    ax.plot(x, ubs, 'b-.', label=r'Upper Bound Signed')
    ax.plot(x, -ubs, 'b-.')
    ax.plot(x, ubst, 'k:', label=r'Upper Bound Signed Trimmed')
    ax.plot(x, -ubst, 'k:')
    ax.legend()
    fig.savefig('l1_signed_plot.png', dpi=300, bbox_inches='tight')

ics = np.arange(len(psqi))
ano = ics[~psqi]
ano_t = ics[~psqi_t]
ano_s = ics[~psqi_s]
ano_st = ics[~psqi_st]

# Plot params
pparam = dict(
    xlabel = r'$x$',
    ylabel = r'$y$',
    xscale = 'linear',
    yscale = 'linear',
    ylim = (y_down.min(), y_up.max()),
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(x, y_hat, '.', color='gray', markersize=0.5, label=r'$\hat{y}$')
    ax.plot(x[ano], y_hat[ano],         'r.', markersize=0.5, label=r'$\hat{y}$')
    #ax.plot(x[ano_t], y_hat[ano_t],     '.', label=r'$\hat{y}_t$')
    #ax.plot(x[ano_s], y_hat[ano_s],     '.', label=r'$\hat{y}_s$')
    #ax.plot(x[ano_st], y_hat[ano_st],   '.', label=r'$\hat{y}_{st}$')
    ax.plot(x, y, 'k-', label=r'$y$')
    ax.plot(x, y_up, 'k--', label=r'$y_{up}$')
    ax.plot(x, y_down, 'k--', label=r'$y_{down}$')
    fig.savefig('ano_plot.png', dpi=300, bbox_inches='tight')

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(x, y_hat, '.', color='gray', markersize=0.5, label=r'$\hat{y}$')
    #ax.plot(x[ano], y_hat[ano],         'r.', markersize=0.5, label=r'$\hat{y}$')
    ax.plot(x[ano_t], y_hat[ano_t],     'g.', markersize=0.5, label=r'$\hat{y}_t$')
    #ax.plot(x[ano_s], y_hat[ano_s],     'b.', markersize=0.5, label=r'$\hat{y}_s$')
    #ax.plot(x[ano_st], y_hat[ano_st],   'k.', markersize=0.5, label=r'$\hat{y}_{st}$')
    ax.plot(x, y, 'k-', label=r'$y$')
    ax.plot(x, y_up, 'k--', label=r'$y_{up}$')
    ax.plot(x, y_down, 'k--', label=r'$y_{down}$')
    fig.savefig('ano_t_plot.png', dpi=300, bbox_inches='tight')

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(x, y_hat, '.', color='gray', markersize=0.5, label=r'$\hat{y}$')
    #ax.plot(x[ano], y_hat[ano],         'r.', markersize=0.5, label=r'$\hat{y}$')
    #ax.plot(x[ano_t], y_hat[ano_t],     'g.', markersize=0.5, label=r'$\hat{y}_t$')
    ax.plot(x[ano_s], y_hat[ano_s],     'b.', markersize=0.5, label=r'$\hat{y}_s$')
    #ax.plot(x[ano_st], y_hat[ano_st],   'k.', markersize=0.5, label=r'$\hat{y}_{st}$')
    ax.plot(x, y, 'k-', label=r'$y$')
    ax.plot(x, y_up, 'k--', label=r'$y_{up}$')
    ax.plot(x, y_down, 'k--', label=r'$y_{down}$')
    fig.savefig('ano_s_plot.png', dpi=300, bbox_inches='tight')

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(x, y_hat, '.', color='gray', markersize=0.5, label=r'$\hat{y}$')
    #ax.plot(x[ano], y_hat[ano],         'r.', markersize=0.5, label=r'$\hat{y}$')
    #ax.plot(x[ano_t], y_hat[ano_t],     'g.', markersize=0.5, label=r'$\hat{y}_t$')
    #ax.plot(x[ano_s], y_hat[ano_s],     'b.', markersize=0.5, label=r'$\hat{y}_s$')
    ax.plot(x[ano_st], y_hat[ano_st], '.', color='purple', markersize=0.5, label=r'$\hat{y}_{st}$')
    ax.plot(x, y, 'k-', label=r'$y$')
    ax.plot(x, y_up, 'k--', label=r'$y_{up}$')
    ax.plot(x, y_down, 'k--', label=r'$y_{down}$')
    fig.savefig('ano_st_plot.png', dpi=300, bbox_inches='tight')
