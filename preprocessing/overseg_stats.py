import numpy as np
from pathlib import Path
import plotly.graph_objects as go


base = Path('preprocessing/scannet_scenes_01_100')
segment_sizes = []
segment_counts = []
for scene_p in base.glob("*.npy"):
    scene = np.load(scene_p)
    segs, cur_seg_cnts = np.unique(scene[:, -3], return_counts=True)
    segment_counts.append(segs.shape[0])
    segment_sizes.extend(list(cur_seg_cnts))

fig = go.Figure(data=[go.Histogram(x=segment_sizes, histnorm='probability',)])
fig2 = go.Figure(data=[go.Histogram(x=segment_counts)])
fig.show()
fig2.show()