#include <algorithm>
#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
//#include "madrona/mesh_bvh2.hpp"
#ifdef MADRONA_GPU_MODE
#include <madrona/mw_gpu/host_print.hpp>
#define LOG(...) mwGPU::HostPrint::log(__VA_ARGS__)
#else
#define LOG(...)
#endif

using namespace madrona;
using namespace madrona::math;

namespace RenderingSystem = madrona::render::RenderingSystem;

namespace madRender {

inline Quat eulerToQuat(float yaw, float pitch) {
    float ex = pitch;
    float ey = 0;
    float ez = yaw;
    float sx = sinf(ex * 0.5f);
    float cx = cosf(ex * 0.5f);
    float sy = sinf(ey * 0.5f);
    float cy = cosf(ey * 0.5f);
    float sz = sinf(ez * 0.5f);
    float cz = cosf(ez * 0.5f);

    ex = (float)(cy * sx * cz - sy * cx * sz);
    ey = (float)(sy * cx * cz + cy * sx * sz);
    ez = (float)(cy * cx * sz - sy * sx * cz);
    float w = (float)(cy * cx * cz + sy * sx * sz);

    Quat cur_rot = Quat{w, ex, ey, ez};
    return cur_rot;
}

// Register all the ECS components and archetypes that will be
// used in the simulation
void Sim::registerTypes(ECSRegistry &registry, const Config &cfg)
{
    base::registerTypes(registry);

    RenderingSystem::registerTypes(registry, cfg.renderBridge);

    registry.registerArchetype<Agent>();
    registry.registerArchetype<DummyRenderable>();
    registry.registerSingleton<TimeSingleton>();

    registry.exportColumn<render::RaycastOutputArchetype,
                          render::RGBOutputBuffer>(
        (uint32_t)ExportID::RaycastRGB);
    registry.exportColumn<render::RaycastOutputArchetype,
                          render::DepthOutputBuffer>(
        (uint32_t)ExportID::RaycastDepth);
    registry.exportColumn<render::RaycastOutputArchetype,
                          render::SegmaskOutputBuffer>(
        (uint32_t)ExportID::RaycastSegmask);

    registry.exportColumn<DummyRenderable, Position>(
        (uint32_t)ExportID::InstancePosition);
    registry.exportColumn<DummyRenderable, Rotation>(
        (uint32_t)ExportID::InstanceRotation);

    registry.exportColumn<Agent, Position>(
        (uint32_t)ExportID::CameraPosition);
    registry.exportColumn<Agent, Rotation>(
        (uint32_t)ExportID::CameraRotation);
}

inline void timeUpdateSys(Engine &ctx,
                          TimeSingleton &time_single)
{
    time_single.currentTime += 0.05f;
}

#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraph::NodeID queueSortByWorld(TaskGraph::Builder &builder,
                                   Span<const TaskGraph::NodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}
#endif

static void setupStepTasks(TaskGraphBuilder &builder, 
                           const Sim::Config &cfg)
{
    // Turn policy actions into movement
    auto time_sys = builder.addToGraph<ParallelForNode<Engine,
         timeUpdateSys,
            TimeSingleton
        >>({});

    auto clear_tmp = builder.addToGraph<ResetTmpAllocNode>({time_sys});
    (void)clear_tmp;

#ifdef MADRONA_GPU_MODE
    // RecycleEntitiesNode is required on the GPU backend in order to reclaim
    // deleted entity IDs.
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({clear_tmp});
    (void)recycle_sys;
#endif

#ifdef MADRONA_GPU_MODE
    // Sort entities, this could be conditional on reset like the second
    // BVH build above.
    auto sort_agents = queueSortByWorld<Agent>(
        builder, {recycle_sys});
    (void)sort_agents;
#endif
}

static void setupRenderTasks(TaskGraphBuilder &builder, 
                             const Sim::Config &)
{
    RenderingSystem::setupTasks(builder, {});
}

// Build the task graph
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &cfg)
{
    setupStepTasks(taskgraph_mgr.init(TaskGraphID::Step), cfg);
    setupRenderTasks(taskgraph_mgr.init(TaskGraphID::Render), cfg);
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &world_init)
    : WorldBase(ctx)
{
    // Initialize the rendering system
    RenderingSystem::init(ctx, cfg.renderBridge);



    // Import the instances
    ImportedInstance *instances = cfg.importedInstances +
                                  world_init.instancesOffset;
    for (uint32_t i = 0; i < world_init.numInstances; ++i) {
        ImportedInstance *inst = &instances[i];

        Entity e_inst = ctx.makeEntity<DummyRenderable>();
        ctx.get<Position>(e_inst) = inst->position;
        ctx.get<Rotation>(e_inst) = inst->rotation;
        ctx.get<Scale>(e_inst) = inst->scale;
        ctx.get<ObjectID>(e_inst).idx = inst->objectID;
        render::RenderingSystem::makeEntityRenderable(ctx, e_inst);
    }

    // Import the cameras
    ImportedCamera *cameras = cfg.importedCameras +
                              world_init.camerasOffset;
    for (uint32_t i = 0; i < world_init.numCameras; ++i) {
        ImportedCamera *cam = &cameras[i];

        Entity agent = ctx.makeEntity<Agent>();

        // Create a render view for the agent
        render::RenderingSystem::attachEntityToView(ctx,
                agent,
                90.f, 0.001f,
                { 0,0,0 });

        ctx.get<Position>(agent) = cam->position;
        ctx.get<Rotation>(agent) = cam->rotation;
    }
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
