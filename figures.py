# %%
# Figure 1, 
import matplotlib.pyplot as plt

# Sample data
epochs = list(range(1, 11))

# Hit@10 values for two methods
# TODO: Replace hit_rate_sl
hit_rate_sl = [0.23155829310417175,
    0.24690373241901398,
    0.24260510504245758,
    0.23605866730213165,
    0.2467593550682068,
    0.24105693399906158,
    0.23594073951244354,
    0.2265627384185791,
    0.22530348598957062,
    0.22092097997665405]
hit_rate_rl = [0.41964617371559143, 0.47415193915367126, 0.48440566658973694, 0.48695698380470276, 0.49150916934013367, 0.49094706773757935, 0.49089542031288147, 0.49704504013061523, 0.495708167552948, 0.49600139260292053]
hit_rate_ppo = [
  0.25609317421913147,
  0.1634155809879303,
  0.06780653446912766,
  0.09893038868904114,
  0.0982380360364914,
  0.06374765187501907,
  0.0966496616601944,
  0.09968721866607666,
  0.13076692819595337,
  0.14944376051425934
]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, hit_rate_sl, marker='o', label='SL', linestyle='-')
plt.plot(epochs, hit_rate_rl, marker='s', label='RL', linestyle='--')
plt.plot(epochs, hit_rate_ppo, marker='^', label='PPO', linestyle='-.')

# Adding titles and labels
plt.title('Epochs vs. Hit Rate@10')
plt.xlabel('Epochs')
plt.ylabel('Hit Rate@10')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Sample data
methods = ['Pop Baseline', 'SL', 'RL', 'PPO']

# Hit rates for the three methods
hit_rate_10 = [0.0933, 0.21089522540569305, 0.407078355550766, 0.14944376]
hit_rate_15 = [0.1221, 0.24636423587799072, 0.4526234567165375, 0.0826490]
hit_rate_20 = [0.1580, 0.23812147974967957, 0.5072090029716492, 0.09574]

# Hit rate categories
hit_rates = ['HitRate@10', 'HitRate@15', 'HitRate@20']

# Data to plot
data = [hit_rate_10, hit_rate_15, hit_rate_20]

# Bar chart parameters
bar_width = 0.2
x = np.arange(len(hit_rates))

# Create figure and axis with high DPI
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# Using "tab20" colormap for better colors
colors = plt.cm.tab20.colors

# Plotting
for i in range(len(methods)):
    ax.bar(x + i * bar_width - bar_width, [data[j][i] for j in range(len(hit_rates))], width=bar_width, color=colors[i], label=methods[i])

# Adding titles and labels
ax.set_xlabel('Hit Rate Category')
ax.set_ylabel('Hit Rate')
ax.set_title('Comparison of Hit Rates by Method on Test Data')
ax.set_xticks(x)
ax.set_xticklabels(hit_rates)

# Adding legend
ax.legend()

# Display the plot
plt.show()
# %%
