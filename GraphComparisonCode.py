import matplotlib.pyplot as plt
import numpy as np

# Replace these lists with the actual data
models = ['RF', 'SVM', 'RF+NB', 'SVM+NB']
accuracies = [93.18, 90.91, 0.88, 0.54]  # Replace with your actual accuracy values
precisions = [1, 1, 0.84, 0.75]  # Replace with your actual precision values
recalls = [0.86, 0.7, 0.95, 0.34]

# Bar chart for Accuracy, Precision, and Recall
bar_width = 0.25
index = np.arange(len(models))

plt.figure(figsize=(10, 6))

plt.bar(index, accuracies, width=bar_width, label='Accuracy')
plt.bar(index + bar_width, precisions, width=bar_width, label='Precision')
plt.bar(index + 2 * bar_width, recalls, width=bar_width, label='Recall')

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Comparison of Model Metrics')
plt.xticks(index + bar_width, models)
plt.legend()
plt.tight_layout()

plt.show()

# Function to plot radar chart with multiple metrics
def plot_multi_metric_radar(model_names, accuracy_values, precision_values, recall_values):
    labels = np.array(model_names)
    num_models = len(labels)

    metrics_values = [
        np.array(accuracy_values),
        np.array(precision_values),
        np.array(recall_values)
    ]
    metrics_labels = ['Accuracy', 'Precision', 'Recall']

    angles = np.linspace(0, 2 * np.pi, num_models, endpoint=False).tolist()
    angles += angles[:1]  # To close the plot

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for i, (metric_label, metric_data) in enumerate(zip(metrics_labels, metrics_values)):
        values = np.concatenate((metric_data, [metric_data[0]]))  # Close the plot

        ax.plot(angles, values, label=metric_label)
        ax.fill(angles, values, alpha=0.3)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.title('Comparison of Model Metrics', size=16, y=1.1)
    plt.show()
