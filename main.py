from user_prompt_to_scene_graph import create_scene_graph, index_scene_graph, enrich_regional_prompt
from scripts.scene_graph_to_bbox_prediction import scene_graph_to_bbox_prediction
from gradio_client import Client
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

user_input = (
    "A man wearing a hat is sitting on a chair beside a table. "
    "On the table, there is a plate. "
)

scene_graph = create_scene_graph(user_input)
scene_graph_index = index_scene_graph(scene_graph)

bounding_box = scene_graph_to_bbox_prediction(scene_graph_index)
regional_prompt = enrich_regional_prompt(user_input, scene_graph)
client = Client("HuiZhang0812/CreatiLayout")
result = client.predict(
		global_caption=user_input,
		box_detail_phrases_list={"headers":["Region Captions"],"data":[[p] for p in regional_prompt],"metadata":None},
		boxes={"headers":["x1","y1","x2","y2"],"data":[bounding_box[i].tolist() for i in range(len(bounding_box))]},
		seed=21,
		randomize_seed=False,
		guidance_scale=7.5,
		num_inference_steps=28,
		api_name="/process_image_and_text"
)

img = Image.open(result[1])
img.show()
image_np = np.array(img)

fig, ax = plt.subplots(1, figsize=(8, 8))
ax.imshow(image_np)
ax.set_title("Bounding Boxes")

# Get image height and width
height, width = image_np.shape[:2]

# Draw each bounding box
for i, (x1, y1, x2, y2) in enumerate(bounding_box):
    rect = patches.Rectangle(
        (x1 * width, y1 * height),             # top-left corner (scaled)
        (x2 - x1) * width, (y2 - y1) * height, # width and height (scaled)
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)
    ax.text(x1 * width, y1 * height - 5, f"Box {i+1}", color='red', fontsize=10, weight='bold')

plt.axis('off')

plt.show()
