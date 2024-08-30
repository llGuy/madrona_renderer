import torch
import matplotlib.pyplot as plt
import madrona_renderer as m

asset_paths = [
    # Cube
    "../data/wall_render.obj",

    # Plane
    "../data/plane.obj"
]

instances = [
    # Cube
    m.ImportedInstance(
        position=[ 0.0, 0.0, 15.0 ],
        rotation=[ 0.707107, 0.707107, 0.0, 0.0 ], # w, x, y, z
        scale=[ 10.0, 10.0, 10.0 ],
        object_id=0
    ),

    # Plane
    m.ImportedInstance(
        position=[ 0.0, 0.0, 0.0 ],
        rotation=[ 0.707107, 0.0, 0.0, 0.707107 ], # w, x, y, z
        scale=[ 0.01, 0.01, 0.01 ],
        object_id=1
    ),
]

cameras = [
    m.ImportedCamera(
        position=[ -30.0, -30.0, 15.0 ],
        rotation=[ 0.999687, 0.024997, 0.000000, 0.024997 ]
    )
]

# Make 16 worlds all with a camera rendering the same thing
num_worlds = 1
world_inits = []

for _ in range(num_worlds):
    world_inits.append(
        m.WorldInit(
            num_instances=2,
            instance_offset=0,
            num_cameras=1,
            camera_offset=0))

renderer = m.MadronaRenderer(
    gpu_id=0,
    num_worlds=num_worlds,
    render_mode=m.RenderMode.Raytracer,
    batch_render_view_width=64,
    batch_render_view_height=64,
    asset_paths=asset_paths,
    instances=instances,
    cameras=cameras,
    worlds=world_inits
)

# Render!

plt.ion()
plt.show()

positions = renderer.instance_position_tensor().to_torch()

for _ in range(16):
    positions[0][2] += 1.0

    renderer.step()
    rgb_tensor = renderer.segmask_tensor().to_torch()
    cpu_tensor = rgb_tensor.cpu()

    plt.imshow(rgb_tensor[0].transpose(0, 1).cpu())
    plt.pause(0.1)
