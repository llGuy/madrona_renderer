import torch
import matplotlib.pyplot as plt
import madrona_renderer as m
import math
import numpy as np


# The textures aren't used by this example but still work
# and use the same API as before.
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
    # Will render the triangle (which is the first mesh)
    m.ImportedInstance(
        position=[ 0.0, 0.0, 15.0 ],
        rotation=[ 0.707107, 0.707107, 0.0, 0.0 ], # w, x, y, z
        scale=[ 10.0, 10.0, 10.0 ],
        object_id=0
    ),
]

cameras = [
    m.ImportedCamera(
        position=[-22.343935, -21.845375, 27.061676],
        rotation=[0.913407, -0.112268, 0.047731, -0.388336]
    )
]

num_worlds = 4
world_inits = []

for _ in range(num_worlds):
    world_inits.append(
        m.WorldInit(
            num_instances=3,
            instance_offset=0,
            num_cameras=1,
            camera_offset=0))



# Manually load your geometry here!
vertices = np.array([
    [ 0.0, 0.0, 0.0 ],
    [ 5.0, 0.0, 10.0 ],
    [ 10.0, 0.0, 0.0 ]
], dtype=np.float32)

# UVs are just 0 because this triangle isn't textured,
# but if your model was textured you'd have to fill this
uvs = np.array([
    [ 0.0, 0.0 ],
    [ 0.0, 0.0 ],
    [ 0.0, 0.0 ]
], dtype=np.float32)

indices = np.array([
    0, 1, 2
], dtype=np.uint32)

# The only offset here is 0 because there is one model.
# If you had more models, you'd have to specify an offset
# for each of your models. The offset would refer the offset
# of the first vertex in the vertices array (this is offset
# of the Vector3, not of the float
mesh_vert_offsets = np.array([
    0
], dtype=np.uint32)

# Same as above, but for the indices
mesh_index_offsets = np.array([
    0
], dtype=np.uint32)

# You need to specify the material of each mesh.
# This one has material -1 meaning I didn't give it a material for now.
# But if you wanted to give this a material, you'd have to pass it the
# index of the material in the AdditionalMaterial array.
mesh_materials = np.array([
    -1
], dtype=np.int32)

renderer = m.MadronaRenderer(
    gpu_id=0,
    num_worlds=num_worlds,
    render_mode=m.RenderMode.Raytracer,
    batch_render_view_width=64,
    batch_render_view_height=64,
    mesh_vertices=vertices,
    mesh_uvs=uvs,
    mesh_indices=indices,
    mesh_vertex_offsets=mesh_vert_offsets,
    mesh_indices_offsets=mesh_index_offsets,
    mesh_materials=mesh_materials,
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

for _ in range(128):
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
