import pickle
from huggingface_hub import hf_hub_download
import plotly.express as px
from pathlib import Path
import h5py
import numpy as np
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import os
from sklearn import manifold, datasets

def find_h5_files(base_path="/home/regmjc3/pillay_lab_server/digital_pathology/processed"):
    """Finds all .h5 files under 'slide_features_titan' subfolders for all IDs."""
    base_path = Path(base_path)
    h5_files = []
    ids = []
    slide_names = []

    for file in base_path.rglob("*/10x_256px_0px_overlap/slide_features_titan/*.h5"):
        h5_files.append(file)
        id_folder = file.parents[2].name  # Extracts the [id] folder name
        ids.append(id_folder)
        slide_names.append(file.stem)

    return h5_files, ids, slide_names

def load_h5_files(h5_files):
    """Loads data from found .h5 files."""
    embeddings = np.empty((0, 768), dtype=np.float32)
    for file in h5_files:
        file = Path(file)  # Ensure it's a Path object
        with h5py.File(file, "r") as f:
            new_embedding = f['features'][()]
            embeddings = np.vstack([embeddings, new_embedding])
    return embeddings

h5_files, ids, slide_names = find_h5_files()
print(f"Found {len(h5_files)} .h5 files")
embeddings = np.empty((0, 768), dtype=np.float32)
data = load_h5_files(h5_files)

# data = load_h5_files(h5_files)
# embeddings = np.empty((0, 768), dtype=np.float32)
# slide_embedding = '/home/regmjc3/pillay_lab_server/digital_pathology/processed/06/10x_256px_0px_overlap/slide_features_titan/500125.h5'
# with h5py.File(slide_embedding, "r") as f:
#     new_embedding = f['features'][()]
#     embeddings = np.vstack([embeddings, f.embeddings])

# Diagnosis mapping
diagnosis_mapping = {
    1: "MPNST",
    2: "Dermatofibrosarcoma protuberans",
    3: "Neurofibroma",
    4: "Nodular fasciitis",
    5: "Desmoid Fibromatosis",
    6: "Synovial sarcoma",
    7: "Lymphoma",
    8: "Glomus tumour",
    9: "Intramuscular myxoma",
    10: "Ewing",
    11: "Schwannoma",
    12: "Myxoid liposarcoma",
    13: "Leiomyosarcoma",
    14: "Solitary fibrous tumour",
    15: "Low grade fibromyxoid sarcoma"
}

# Create DataFrame for visualisation
df_tsne = pd.DataFrame({
    'tSNE1': data[:, 0],
    'tSNE2': data[:, 1],
    'slide_id': slide_names,
    'Diagnosis': [diagnosis_mapping[int(label)] for label in ids]  # Map numerical labels to diagnosis names
})
# Define color palette using plotly's categorical colors
unique_labels = df_tsne['Diagnosis'].unique()
color_map = px.colors.qualitative.Plotly
label_to_color = {label: color_map[i % len(color_map)] for i, label in enumerate(unique_labels)}

# Save interactive plot
fig = px.scatter(
    df_tsne,
    x='tSNE1',
    y='tSNE2',
    color=df_tsne['Diagnosis'],  # Color by Diagnosis names
    hover_data=['slide_id', 'Diagnosis'],  # Show case ID on hover
    title="Interactive t-SNE Visualisation of TITAN Slide-Level Embeddings (Grouped by Ground Truth)",
    labels={'color': 'Diagnosis'},  # Change legend title to 'Diagnosis'
    width=1200, height=800,
    color_discrete_map=label_to_color
)
output_path = '/home/regmjc3/SARC_classifier'
output_html = os.path.join(output_path, "tsne_visualisation_interactive.html")
fig.write_html(output_html)
print(f"Interactive t-SNE visualisation saved at: {output_html}")

t0 = time()
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
Y = tsne.fit_transform(data['embeddings'])
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))
plt.scatter(Y[:, 0], Y[:, 1])

with open(os.path.join(output_path, "tsne_visualisation_interactive.html"), "r", encoding="utf-8") as f:
    html_content = f.read()

########## below JavaScript code is provided by Jianan Chen for improve user experience ##########
# JavaScript code
js_code = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    var plot = document.getElementsByClassName('plotly-graph-div')[0];

    plot.on('plotly_click', function(data) {
        if (data.points.length > 0) {
            var filename = data.points[0].customdata[0]; 
            navigator.clipboard.writeText(filename).then(function() {
                alert("Copied to clipboard: " + filename);
            }).catch(function(err) {
                console.error("Failed to copy: ", err);
            });
        }
    });
});
</script>
"""

# Insert the JavaScript code before </body>
html_content = html_content.replace("</body>", js_code + "\n</body>")

# Save the modified HTML
with open(os.path.join(output_path, "tsne_visualisation_interactive.html"), "w", encoding="utf-8") as f:
    f.write(html_content)

print(1)