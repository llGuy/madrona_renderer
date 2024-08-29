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

namespace madEscape {

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

    registry.registerComponent<Action>();
    registry.registerComponent<AgentCamera>();
    registry.registerArchetype<Agent>();
    registry.registerArchetype<DummyRenderable>();
    registry.registerSingleton<TimeSingleton>();

    registry.exportColumn<Agent, Action>(
        (uint32_t)ExportID::Action);
    registry.exportColumn<render::RaycastOutputArchetype,
                          render::RGBOutputBuffer>(
        (uint32_t)ExportID::Raycast);
}

// #define DYNAMIC_MOVEMENT

inline void movementSystem(Engine &ctx,
                           Entity e,
                           Action &action, 
                           Rotation &rot,
                           Position &pos,
                           AgentCamera& cam)
{
    Quat cur_rot = eulerToQuat(cam.yaw, 0);
    
    int actionX = action.x - 1;
    int actionY = action.y - 1;

    Vector3 walk_vec = cur_rot.rotateVec(
            { (float)actionY, (float)actionX, 0});

    walk_vec = walk_vec.length2() == 0 ? 
        Vector3{0,0,0} : walk_vec.normalize();

    walk_vec *= 0.8f;

    Vector3 new_velocity = {0,0,0};

    new_velocity.x =  walk_vec.x;
    new_velocity.y = walk_vec.y;
    new_velocity.z = action.z - 1;

    float range = 5.0f;
    float random_thing =  
        ctx.data().rng.sampleUniform() * range - range / 2.f;

#if defined(DYNAMIC_MOVEMENT)
    cam.yaw += 0.15f;
    cam.pitch = -0.33f;
#endif

    cam.yaw += (action.rot - 1) * consts::sensitivity;
    cam.yaw -= math::pi_m2 * 
               std::floor((cam.yaw + math::pi) * 
               (1.0f / math::pi_m2));

    cam.pitch += (action.vrot - 1) * consts::sensitivity;
    cam.pitch = std::clamp(cam.pitch, 
                           -math::pi_d2,
                           math::pi_d2);

    pos += new_velocity;


#if defined(DYNAMIC_MOVEMENT)
    float current_time = ctx.singleton<TimeSingleton>().currentTime;
    float entity_offset = (float)e.id;
    pos = Vector3{ 
        20.f * std::cosf(random_thing + 
                         current_time +
                         entity_offset), 

        20.f * std::sinf(random_thing + 
                         current_time +
                         entity_offset),

        15.f + std::sinf(random_thing + 
                         current_time +
                         entity_offset) * 3.0f
    };

    pos.x += ctx.data().worldCenter.x;
    pos.y += ctx.data().worldCenter.y;
#endif

    rot = eulerToQuat(cam.yaw, cam.pitch);
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
    auto move_sys = builder.addToGraph<ParallelForNode<Engine,
        movementSystem,
            Entity,
            Action,
            Rotation,
            Position,
            AgentCamera
        >>({});

    auto time_sys = builder.addToGraph<ParallelForNode<Engine,
         timeUpdateSys,
            TimeSingleton
        >>({move_sys});

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
                             const Sim::Config &cfg)
{
    RenderingSystem::setupTasks(builder, {});
}

// Build the task graph
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &cfg)
{
    setupStepTasks(taskgraph_mgr.init(TaskGraphID::Step), cfg);
    setupRenderTasks(taskgraph_mgr.init(TaskGraphID::Render), cfg);
}

static void loadInstances(Engine &ctx)
{
    for (int i = 0; i < (int)ctx.data().numImportedInstances; ++i) {
        ImportedInstance *imp_inst = &ctx.data().importedInstances[i];

        Entity e_inst = ctx.makeEntity<DummyRenderable>();
        ctx.get<Position>(e_inst) = imp_inst->position;
        ctx.get<Rotation>(e_inst) = imp_inst->rotation;
        ctx.get<Scale>(e_inst) = imp_inst->scale;
        ctx.get<ObjectID>(e_inst).idx = imp_inst->objectID;

        render::RenderingSystem::makeEntityRenderable(ctx, e_inst);
    }

    { // Create the agent entity of this world
        Entity agent = ctx.data().agent =
            ctx.makeEntity<Agent>();

        ctx.get<AgentCamera>(agent) = { 
            .yaw = 0,
            .pitch = 0 
        };

        // Create a render view for the agent
        render::RenderingSystem::attachEntityToView(ctx,
                agent,
                90.f, 0.001f,
                { 0,0,0 });

        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };

        ctx.get<Rotation>(agent) = Quat::angleAxis(
            math::pi/2.f,
            math::up);

        ctx.get<Action>(agent) = Action {
            0, 0, 0, 0, 1, 1, 1, 1, 1
        };
    }
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    ctx.data().initRandKey = cfg.initRandKey;
    ctx.data().rng = RNG(rand::split_i(ctx.data().initRandKey,
        0, (uint32_t)ctx.worldID().idx));

    uint32_t current_scene = ctx.data().rng.sampleI32(
            0, cfg.numUniqueScenes);

    UniqueScene *unique_scene = &cfg.uniqueScenes[current_scene];

    importedInstances = cfg.importedInstances + 
        unique_scene->instancesOffset;

    numImportedInstances = unique_scene->numInstances;

    worldCenter = { unique_scene->center.x, unique_scene->center.y };

    RenderingSystem::init(ctx, cfg.renderBridge);

    loadInstances(ctx);

    ctx.singleton<TimeSingleton>().currentTime = 0.f;
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
