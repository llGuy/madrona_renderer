import madrona_renderer as m

asset_paths = [
    # Cube
    "/home/luc/Development/madrona_renderer/data/wall_render.obj",

    # Plane
    "/home/luc/Development/madrona_renderer/data/plane.obj"
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
        rotation=[ 0.0, 0.0, 0.0, 0.707107 ]
    )
]

# Make 16 worlds all with a camera rendering the same thing
num_worlds = 16
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
renderer.step()
