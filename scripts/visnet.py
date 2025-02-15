import scenparse.SumoNetVis
import matplotlib.pyplot as plt
# Plot Sumo Network
import os

# /media/led/WD_2TB/Dataset/maps/waymo-validation
for i, directory in enumerate(os.listdir("/home/led/Documents/maps/")):

    if i >= 10:
        break

    net = scenparse.SumoNetVis.Net(f"/home/led/Documents/maps/{directory}/{directory}.net.xml")
    net.plot(clip_to_limits=False, zoom_to_extents=True, style="USA", plot_stop_lines=False,)
    # Show figure
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().axis('off')  # Hide the axis
    plt.gcf().set_size_inches(18.5, 10.5)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove empty padding
    plt.savefig(f"outputs/img-sumo/{directory}.png", dpi=300, bbox_inches='tight', pad_inches=0)