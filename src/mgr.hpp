#include "sim.hpp"

#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include <madrona/render/render_mgr.hpp>

namespace madRender {

// The Manager class encapsulates the linkage between the outside training
// code and the internal simulation state (src/sim.hpp / src/sim.cpp)
//
// Manager is responsible for initializing the simulator, loading physics
// and rendering assets off disk, and mapping ECS components to tensors
// for learning
class Manager {
public:
    struct Config {
        madrona::ExecMode execMode; // CPU or CUDA
        int gpuID; // Which GPU for CUDA backend?
        uint32_t numWorlds; // Simulation batch size
        uint32_t batchRenderViewWidth = 64;
        uint32_t batchRenderViewHeight = 64;
        madrona::render::APIBackend *extRenderAPI = nullptr;
        madrona::render::GPUDevice *extRenderDev = nullptr;
        uint32_t raycastOutputResolution = 64;
        bool headlessMode = false;
    };

    Manager(const Config &cfg);
    ~Manager();

    void step();

    void configureAssets(const char **paths,
                         size_t num_paths);

    // This is just going to be a big list of instances to render
    void configureInstances(
            const madRender::ImportedInstance *imported_instances,
            size_t num_instances);

    void configureCameras(const madRender::Camera *cameras,
                          size_t num_cameras);

    void configureWorlds(const madRender::World *worlds);

    // These functions export Tensor objects that link the ECS
    // simulation state to the python bindings / PyTorch tensors (src/bindings.cpp)
    madrona::py::Tensor actionTensor() const;
    madrona::py::Tensor rgbTensor() const;
    madrona::py::Tensor depthTensor() const;
    madrona::py::Tensor raycastTensor() const;

    // These functions are used by the viewer to control the simulation
    // with keyboard inputs in place of DNN policy actions
    void setAction(int32_t world_idx,
                   int32_t agent_idx,
                   int32_t move_amount,
                   int32_t move_angle,
                   int32_t rotate,
                   int32_t grab,
                   int32_t x,
                   int32_t y,
                   int32_t z,
                   int32_t rot,
                   int32_t vrot);

    madrona::render::RenderManager & getRenderManager();

    uint32_t numAgents;

private:
    struct RenderConfig {
        // Paths to load assets from
        std::vector<std::string> paths;

        ImportedInstance *importedInstancesGPU;
        size_t numImportedInstances;

        Camera *camerasGPU;
        size_t numCameras;

        World *worldsGPU;
        size_t numWorlds;
    };

    struct Impl;
    struct CUDAImpl;

    std::unique_ptr<Impl> impl_;

    RenderConfig render_cfg_;
};

}
