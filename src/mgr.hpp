#include "sim.hpp"

#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include <madrona/importer.hpp>
#include <madrona/render/render_mgr.hpp>

namespace madRender {

// Additional materials / textures / ...
using AdditionalMaterial = madrona::imp::SourceMaterial;

struct ImportedAsset {
    std::string path;

    // Optional
    int32_t matID;
};

// The Manager class encapsulates the linkage between the outside training
// code and the internal simulation state (src/sim.hpp / src/sim.cpp)
//
// Manager is responsible for initializing the simulator, loading physics
// and rendering assets off disk, and mapping ECS components to tensors
// for learning
class Manager {
public:
    enum class RenderMode {
        Rasterizer,
        Raytracer
    };

    struct GeometryConfig {
        const madrona::math::Vector3 *vertices;
        const madrona::math::Vector2 *uvs;
        const uint32_t *indices;
        const uint32_t *meshVertexOffsets;
        const uint32_t *meshIndexOffsets;
        const int32_t *meshMaterials;

        uint32_t numVertices;
        uint32_t numIndices;
        uint32_t numMeshes;
    };

    struct Config {
        int gpuID;

        uint32_t numWorlds;

        RenderMode renderMode;

        uint32_t batchRenderViewWidth = 64;
        uint32_t batchRenderViewHeight = 64;

        madrona::render::APIBackend *extRenderAPI = nullptr;
        madrona::render::GPUDevice *extRenderDev = nullptr;

        bool headlessMode = false;

        // Render config
        struct RenderConfig {
            GeometryConfig geoCfg;

            const AdditionalMaterial *additionalMats;
            uint32_t numAdditionalMats;

            const char **additionalTextures;
            uint32_t numAdditionalTextures;

            ImportedInstance *importedInstances;
            uint32_t numInstances;

            ImportedCamera *cameras;
            uint32_t numCameras;

            Sim::WorldInit *worlds;
        } rcfg;
    };

    Manager(const Config &cfg);
    ~Manager();

    void step();

    // These functions export Tensor objects that link the ECS
    // simulation state to the python bindings / PyTorch tensors (src/bindings.cpp)
    madrona::py::Tensor rgbTensor() const;
    madrona::py::Tensor depthTensor() const;
    madrona::py::Tensor segmaskTensor() const;
    
    madrona::py::Tensor instancePositionTensor() const;
    madrona::py::Tensor instanceRotationTensor() const;

    madrona::py::Tensor cameraPositionTensor() const;
    madrona::py::Tensor cameraRotationTensor() const;
    
    uint64_t rgbCudaPtr() const;
    uint64_t depthCudaPtr() const;
    uint64_t segmaskCudaPtr() const;

    madrona::render::RenderManager & getRenderManager();

    uint32_t numAgents;

private:
    struct Impl;
    struct CUDAImpl;

    std::unique_ptr<Impl> impl_;
};

}
