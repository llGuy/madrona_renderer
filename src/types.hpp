#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/rand.hpp>
#include <madrona/render/ecs.hpp>

#include "consts.hpp"

namespace madEscape {

// Include several madrona types into the simulator namespace for convenience
using madrona::Entity;
using madrona::RandKey;
using madrona::CountT;
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;

// Discrete action component. Ranges are defined by consts::numMoveBuckets (5),
// repeated here for clarity
struct Action {
    int32_t moveAmount; // [0, 3]
    int32_t moveAngle; // [0, 7]
    int32_t rotate; // [-2, 2]
    int32_t grab; // 0 = do nothing, 1 = grab / release
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t rot;
    int32_t vrot;
};

struct AgentCamera {
    float yaw;
    float pitch;
};

// Entity that is attached to the camera
struct Agent : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    Action,
    AgentCamera,
    madrona::render::RenderCamera
> {};

// Dummy renderable entity (used for imported instances)
struct DummyRenderable : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID,
    madrona::render::Renderable
 > {};

}
