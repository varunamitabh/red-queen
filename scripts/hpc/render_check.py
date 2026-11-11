"""Does MuJoCo render headless here, and does PyTorch see the GPU?"""
import os
import time

import mujoco
import numpy as np
import torch

print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
xml = """<mujoco><worldbody><light pos="0 0 3"/><geom type="plane" size="1 1 .1" rgba=".8 .8 .8 1"/>
<body pos="0 0 .2"><freejoint/><geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/></body></worldbody></mujoco>"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 256, 256)
t0 = time.time()
for _ in range(100):
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    img = renderer.render()
dt = time.time() - t0
assert img.shape == (256, 256, 3) and np.ptp(img) > 0, "rendered a blank frame"
print(f"render OK: {100 / dt:.0f} frames/s at 256x256, pixel range {img.min()}..{img.max()}")
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
