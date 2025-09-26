import json
import matplotlib.pyplot as plt
import time
import numpy as np
import argparse

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        # print(f'{file.readlines()[1] = }')
        # print(f'{file.readlines()[1] = }')
        data = [json.loads(line) for line in file.readlines()]
    return data

def get_loss_values(data):
    # return [data[i]["mIoU"] for i in range(len(data)) if (i != 0 and data[i]["mode"] == "val")]
    return [data[i]["loss"] for i in range(len(data)) if (i != 0 and data[i]["mode"] == "train")]

def plot_loss_values(loss_values):
    plt.ion()  # Turn on interactive mode
    fig = plt.figure()  # Create a new figure
    ax = fig.add_subplot(111)  # Add a subplot to the figure
    line1, = ax.plot(loss_values, 'r-')  # Create a line object for the plot
    plt.show()  # Show the plot

    return fig, ax, line1

def update_plot(fig, ax, line1, new_loss_values):
    line1.set_ydata(new_loss_values)
    ax.relim()
    ax.autoscale_view(True,True,True)
    fig.canvas.draw()
    plt.pause(0.1)  # Pause to allow the plot to update

def main(
    height,
    display_every_nth_value,
    ignore_first_n_values,
    samples_per_entry,
    crop_percentile
    ):

    file_path = '/proj-soils/data/heig-vd_logs_checkpoints/mask2former_beit_adapter_large_512_16k_proj-soils_ss/20240214_110056.log.json'
    # read_interval = 10  # Read the file every 10 seconds
    
    data = read_json_file(file_path)
    loss_values = get_loss_values(data)
    # print(f'{loss_values = }')

    loss_values = loss_values[ignore_first_n_values:]
    loss_values = np.array([loss_values[i] for i in range(len(loss_values)) if i % display_every_nth_value == 0])
    print(loss_values)
    
    print("\n" + "#" * 7 * (len(loss_values)+2))
    print(f'#\n# Plotting loss values from {file_path}\n#')

    quantile = np.quantile(loss_values, crop_percentile)
    loss_values_clipped = loss_values.clip(0, quantile)

    max_value = max(loss_values_clipped)
    min_value = min(loss_values_clipped)
    range_value = max_value - min_value

    # Normalize the values to range from 0 to 40
    normalized_values = [(value - min_value) / range_value * height for value in loss_values_clipped]
    clip_indicator_ar = np.zeros(len(normalized_values), dtype=int) - 999



    # Print the plot
    for i in range(height+2, -1, -1):
        line = '#  '
        for loss_idx, value in enumerate(normalized_values):
            if loss_values[loss_idx] > quantile and clip_indicator_ar[loss_idx] == -999:
                clip_indicator_ar[loss_idx] = i
            if clip_indicator_ar[loss_idx] in [i+8, i+9]:
                line += '  ...  '
            elif int(value) == i:
                line += f"{loss_values[loss_idx]:.1e}"
            elif int(value) > i:
                line += '|-----|'
            else: 
                line += '       '
        
        print(line + "         #")

    print("#   " + ("-" * ((len(normalized_values) * 7) - 2)) + "          #") 

    x_ticks = "#  "
    for i in range(len(normalized_values)):
        x_ticks += f"{int(i*samples_per_entry*display_every_nth_value) + (ignore_first_n_values * samples_per_entry):^7}"
    print(x_ticks  + 'Samples  #')
    print("#\n" + "#" * 7 * (len(loss_values)+2))

if __name__ == "__main__":
    height = 80
    display_every_nth_value = 2
    ignore_first_n_values = 0
    samples_per_entry = 50
    crop_percentile = 0.975

    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="This script plots the loss values from a log file.")
    parser.add_argument('-hl', '--height', help='The height of the plot in characters. Default is 80.', type=int, default=80)
    parser.add_argument('-n', '--display_every_nth_value', help='Display every nth value. Default is 2.', type=int, default=2)
    parser.add_argument('-i', '--ignore_first_n_values', help='Ignore the first n values. Default is 0.', type=int, default=0)
    parser.add_argument('-s', '--samples_per_entry', help='Number of samples per entry. Default is 50.', type=int, default=50)
    parser.add_argument('-c', '--crop_percentile', help='The percentile to crop the values at. Default is 0.975.', type=float, default=0.975)

    args = parser.parse_args()
    main(
        args.height,
        args.display_every_nth_value,
        args.ignore_first_n_values,
        args.samples_per_entry,
        args.crop_percentile
    )