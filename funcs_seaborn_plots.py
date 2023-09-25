import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plot TraCE score heatmap with increased readability
def plot_trace_heatmap(trace_countries_df, SSPs, cumulative_method="sum",
                       figsize=(8, 8), cmap="YlOrRd", cbar=True, format=".2f",
                       title="", xlabel="", ylabel="", font_scale=1.5):

    # Set the font scale
    sns.set(font_scale=font_scale)
    
    # Melt the DataFrame
    df = trace_countries_df.melt(id_vars=['code', 'name'], value_vars=[f"SSP{ssp}" for ssp in SSPs], var_name='SSP', value_name='TraCE Score')
    
    # Create the heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(data=df.pivot(index="name", columns="SSP", values="TraCE Score"),
                     annot=True, cmap=cmap, cbar=cbar, fmt=format, annot_kws={"weight": "bold"})

    # Add a label to the color bar
    if cbar:
        cbar = ax.collections[0].colorbar
        if cumulative_method == "sum":
            description = f"Cumulative TraCE score"
        elif cumulative_method == "average" or "mean":
            description = f"Average TraCE score"
        cbar.set_label(description, fontsize=font_scale * 12)  # Adjust colorbar label font size

    # Customise the plot
    ax.xaxis.tick_top()
    ax.set_title(title, fontsize=font_scale * 14)  # Adjust title font size
    ax.set_xlabel(xlabel, fontsize=font_scale * 12)  # Adjust xlabel font size
    ax.set_ylabel(ylabel, fontsize=font_scale * 12)  # Adjust ylabel font size
    
    plt.tight_layout()
    plt.show()



# Plot TraCE score heatmap
def plot_trace_features_heatmap(trace_features_df, SSPs, cumulative_method="sum", 
                       figsize=(8,5), cmap="YlOrRd", cbar=True, format=".2f", 
                       title="", xlabel="", ylabel="", font_scale=1.5):
    # Set the font scale
    sns.set(font_scale=font_scale)
    
    # Melt the DataFrame
    df = trace_features_df.melt(id_vars=["feature"], value_vars=[f"SSP{ssp}" for ssp in SSPs], var_name='SSP', value_name='TraCE Score')
    
    # Create the heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(data=df.pivot(index="feature", columns="SSP", values="TraCE Score"), 
                     annot=True, cmap=cmap, cbar=cbar, fmt=format, annot_kws={"weight": "bold"})

    
    # Add a label to the color bar
    if cbar:
        cbar = ax.collections[0].colorbar
        if cumulative_method == "sum":
            description = f"Cumulative TraCE score"
        elif cumulative_method == "average" or "mean":
            description = f"Average TraCE score"
        cbar.set_label(description, fontsize=font_scale * 10)

    # Customise the plot
    ax.xaxis.tick_top()
    ax.set_title(title, fontsize=font_scale * 14)
    ax.set_xlabel(xlabel, fontsize=font_scale * 12)
    ax.set_ylabel(ylabel, fontsize=font_scale * 12)
    plt.show()