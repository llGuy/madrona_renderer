import torch
import matplotlib.pyplot as plt
import madrona_renderer as m
import math

asset_paths = [
    # Cube
    m.ImportedAsset(
        path="../data/cube.obj",
        mat_id=0
    ),

    # Plane
    m.ImportedAsset(
        "../data/plane.obj",

        # No material
        mat_id=-1
    )
]

additional_mats = [
    m.AdditionalMaterial(
        color=[ 1, 1, 1, 1 ],
        texture_id=0,
        roughness=0.8,
        metalness=0.2
    )
]

texture_paths = [
    "../data/cube.png"
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
        position=[-22.343935, -21.845375, 27.061676],
        rotation=[0.913407, -0.112268, 0.047731, -0.388336]
    )
]

# Make 16 worlds all with a camera rendering the same thing
num_worlds = 4
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
    materials=additional_mats,
    texture_paths=texture_paths,
    cameras=cameras,
    worlds=world_inits
)

# Render!

plt.ion()
plt.show()

positions = renderer.instance_position_tensor().to_torch()

grid_height = math.ceil(math.sqrt(num_worlds))
grid_width = math.ceil(num_worlds / grid_height)

fig, axes = plt.subplots(grid_width, grid_height, figsize=(10, 10))

for _ in range(32):
    positions[0][2] += 1.0
    positions[1][2] += 2.0
    positions[2][2] += 1.5
    positions[3][2] += 0.5

    renderer.step()
    segmask_tensor = renderer.rgb_tensor().to_torch()
    cpu_tensor = segmask_tensor.cpu()

    for y in range(grid_height):
        for x in range(grid_width):
            image_idx = x + y * grid_width

            if image_idx < num_worlds:
                ax = axes[x, y]
                ax.imshow(cpu_tensor[image_idx].transpose(0, 1))
                ax.axis('off')

    plt.pause(0.1)
