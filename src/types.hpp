#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/rand.hpp>
#include <madrona/render/ecs.hpp>

namespace madRender {

// Include several madrona types into the simulator namespace for convenience
using madrona::Entity;
using madrona::RandKey;
using madrona::CountT;
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;

// Entity that is attached to the camera
struct Agent : public madrona::Archetype<
    Position,
    Rotation,
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
