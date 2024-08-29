#pragma once

// #define MERGE_ALL

#include <madrona/taskgraph_builder.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/rand.hpp>

#include "consts.hpp"
#include "types.hpp"
#include "madrona/mesh_bvh.hpp"


namespace madEscape {

class Engine;

// This enum is used by the Sim and Manager classes to track the export slots
// for each component exported to the training code.
enum class ExportID : uint32_t {
    Action,
    Raycast,
    NumExports,
};

struct ImportedInstance {
    madrona::math::Vector3 position;
    madrona::math::Quat rotation;
    madrona::math::Diag3x3 scale;
    int32_t objectID;
};

enum class TaskGraphID : uint32_t {
    Step,
    Render,
    NumTaskGraphs,
};

// This is used for generic rendering objects
using SimObject = uint32_t;

struct UniqueScene {
    uint32_t numInstances;
    uint32_t instancesOffset;
    uint32_t numObjects;
    madrona::math::Vector3 center;
};

struct TimeSingleton {
    float currentTime;
};

// The Sim class encapsulates the per-world state of the simulation.
// Sim is always available by calling ctx.data() given a reference
// to the Engine / Context object that is passed to each ECS system.
//
// Per-World state that is frequently accessed but only used by a few
// ECS systems should be put in a singleton component rather than
// in this class in order to ensure efficient access patterns.
struct Sim : public madrona::WorldBase {
    struct Config {
        bool autoReset;
        RandKey initRandKey;
        const madrona::render::RenderECSBridge *renderBridge;

        uint32_t numObjects;
        uint32_t numImportedInstances;
        ImportedInstance *importedInstances;

        madrona::math::Vector2 sceneCenter;

        uint32_t numUniqueScenes;
        UniqueScene *uniqueScenes;

        uint32_t numWorlds;

        uint32_t numAgents;

        bool mergeAll;
    };

    struct WorldInit {};

    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static void setupTasks(madrona::TaskGraphManager &taskgraph_mgr,
                           const Config &cfg);

    Sim(Engine &ctx,
        const Config &cfg,
        const WorldInit &);

    madrona::RandKey initRandKey;
    madrona::RNG rng;

    ImportedInstance *importedInstances;
    uint32_t numImportedInstances;

    madrona::math::Vector2 worldCenter;

    madrona::Entity agent;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
public:
    using CustomContext::CustomContext;

    // These are convenience helpers for creating renderable
    // entities when rendering isn't necessarily enabled
    template <typename ArchetypeT>
    inline madrona::Entity makeRenderableEntity();
    inline void destroyRenderableEntity(Entity e);
};

}

#include "sim.inl"
