#include "mgr.hpp"
#include "sim.hpp"

#include <random>
#include <numeric>
#include <algorithm>

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/render/api.hpp>
#include <madrona/physics_assets.hpp>

#include <array>
#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#include <madrona/render/asset_processor.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

#include <madrona_ktx.h>

#define MADRONA_VIEWER

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::render;
using namespace madrona::py;

namespace madEscape {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};


static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg)
{
    if (!mgr_cfg.headlessMode) {
        if (mgr_cfg.extRenderDev || !mgr_cfg.enableBatchRenderer) {
            return Optional<RenderGPUState>::none();
        }
    }

    auto render_api_lib = render::APIManager::loadDefaultLib();
    render::APIManager render_api_mgr(render_api_lib.lib());
    render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

    return RenderGPUState {
        .apiLib = std::move(render_api_lib),
        .apiMgr = std::move(render_api_mgr),
        .gpu = std::move(gpu),
    };
}

static inline Optional<render::RenderManager> initRenderManager(
    const Manager::Config &mgr_cfg,
    const Optional<RenderGPUState> &render_gpu_state)
{
    if (mgr_cfg.headlessMode && !mgr_cfg.enableBatchRenderer) {
        return Optional<render::RenderManager>::none();
    }

    if (!mgr_cfg.headlessMode) {
        if (!mgr_cfg.extRenderDev && !mgr_cfg.enableBatchRenderer) {
            return Optional<render::RenderManager>::none();
        }
    }

    render::APIBackend *render_api;
    render::GPUDevice *render_dev;

    if (render_gpu_state.has_value()) {
        render_api = render_gpu_state->apiMgr.backend();
        render_dev = render_gpu_state->gpu.device();
    } else {
        render_api = mgr_cfg.extRenderAPI;
        render_dev = mgr_cfg.extRenderDev;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = mgr_cfg.enableBatchRenderer,
        .renderMode = render::RenderManager::Config::RenderMode::RGBD,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = consts::maxAgents,
        .maxInstancesPerWorld = 1024,
        .execMode = mgr_cfg.execMode,
        .voxelCfg = {},
    });
}

struct Manager::Impl {
    Config cfg;
    Action *agentActionsBuffer;
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;
    uint32_t raycastOutputResolution;
    bool headlessMode;

    inline Impl(const Manager::Config &mgr_cfg,
                Action *action_buffer,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr,
                uint32_t raycast_output_resolution)
        : cfg(mgr_cfg),
          agentActionsBuffer(action_buffer),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr)),
          raycastOutputResolution(raycast_output_resolution),
          headlessMode(mgr_cfg.headlessMode)
    {}

    inline virtual ~Impl() {}

    virtual void run() = 0;

    virtual Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dimensions) const = 0;

    static inline Impl * init(const Config &cfg);
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph stepGraph;
    MWCudaLaunchGraph renderSetupGraph;
    Optional<MWCudaLaunchGraph> renderGraph;

    inline CUDAImpl(const Manager::Config &mgr_cfg,
                   Action *action_buffer,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   MWCudaExecutor &&gpu_exec,
                   MWCudaLaunchGraph &&step_graph,
                   MWCudaLaunchGraph &&render_setup_graph,
                   Optional<MWCudaLaunchGraph> &&render_graph)
        : Impl(mgr_cfg,
               action_buffer,
               std::move(render_gpu_state), std::move(render_mgr),
               mgr_cfg.raycastOutputResolution),
          gpuExec(std::move(gpu_exec)),
          stepGraph(std::move(step_graph)),
          renderSetupGraph(std::move(render_setup_graph)),
          renderGraph(std::move(render_graph))
    {}

    inline virtual ~CUDAImpl() final {}

    inline virtual void run()
    {
        gpuExec.run(stepGraph);
        gpuExec.run(renderSetupGraph);

        if (renderGraph.has_value()) {
            gpuExec.run(*renderGraph);
        }
    }

    virtual inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};
#else
static_assert(false, "This only works with the CUDA backend");
#endif

static Optional<imp::SourceTexture> ktxImageImportFn(
        void *data, size_t num_bytes)
{
    ktx::ConvertedOutput converted = {};
    ktx::loadKTXMem(data, num_bytes, &converted);

    return imp::SourceTexture {
        .data = converted.textureData,
        .format = imp::SourceTextureFormat::BC7,
        .width = (uint32_t)converted.width,
        .height = (uint32_t)converted.height,
        .numBytes = converted.bufferSize
    };
}

struct LoadResult {
    std::vector<ImportedInstance> importedInstances;
    std::vector<UniqueScene> uniqueSceneInfos;
};


static imp::ImportedAssets loadGLB(
        Optional<render::RenderManager> &render_mgr,
        const std::string &glb_path,
        LoadResult &load_result)
{
    std::vector<std::string> render_asset_paths;

    // Object 0 is the glb object
    render_asset_paths.push_back(
            std::filesystem::path(DATA_DIR) / glb_path);

    // Object 1 is the plane
    render_asset_paths.push_back(
            std::filesystem::path(DATA_DIR) / "plane.obj");

    printf("GLB path to render: %s\n", render_asset_paths[0].c_str());

    load_result.importedInstances.push_back({
        .position = { 0.f, 0.f, 0.f },
        .rotation = Quat::angleAxis(pi_d2, { 1.f, 0.f, 0.f }),
        .scale = Diag3x3{ 10.f, 10.f, 10.f },
        .objectID = 0
    });

    load_result.importedInstances.push_back({
        .position = { 0.f, 0.f, 0.f },
        .rotation = Quat::angleAxis(pi_d2, { 0.f, 0.f, 1.f }),
        .scale = Diag3x3{ 0.01f, 0.01f, 0.01f },
        .objectID = 1
    });

    load_result.uniqueSceneInfos.push_back({
        2, 0, 2, { 0.f, 0.f, 0.f }
    });

    std::vector<const char *> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++)
        render_asset_cstrs.push_back(render_asset_paths[i].c_str());

    imp::AssetImporter importer;

    // Setup importer to handle KTX images
    imp::ImageImporter &img_importer = importer.imageImporter();
    img_importer.addHandler("ktx2", ktxImageImportFn);

    std::array<char, 1024> import_err;
    auto render_assets = importer.importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()),
        true);

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    render_assets->materials.push_back({
        .color = { 1.f, 1.f, 1.f, 1.f },
        .textureIdx = -1,
        .roughness = 1.f,
        .metalness = 0.1f,
    });

    render_assets->objects[1].meshes[0].materialIDX = 
        render_assets->materials.size() - 1;

    if (render_mgr.has_value()) {
        render_mgr->loadObjects(render_assets->objects, 
                render_assets->materials,
                render_assets->textures);

        render_mgr->configureLighting({
            { true, math::Vector3{1.0f, -1.0f, -0.05f}, 
              math::Vector3{1.0f, 1.0f, 1.0f} }
        });
    }

    return std::move(*render_assets);
}

Manager::Impl * Manager::Impl::init(
    const Manager::Config &mgr_cfg)
{
    Sim::Config sim_cfg;
    sim_cfg.autoReset = mgr_cfg.autoReset;
    sim_cfg.initRandKey = rand::initKey(mgr_cfg.randSeed);

    const char *num_agents_str = getenv("HIDESEEK_NUM_AGENTS");
    if (num_agents_str) {
        uint32_t num_agents = std::stoi(num_agents_str);
        sim_cfg.numAgents = num_agents;
    } else {
        sim_cfg.numAgents = 1;
    }

    switch (mgr_cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        std::vector<ImportedInstance> imported_instances;

        sim_cfg.mergeAll = false;

        LoadResult load_result = {};

        auto imported_assets = loadGLB(
                render_mgr,
                mgr_cfg.glbPath,
                load_result);

        sim_cfg.importedInstances = (ImportedInstance *)cu::allocGPU(
                sizeof(ImportedInstance) *
                load_result.importedInstances.size());

        sim_cfg.numImportedInstances = load_result.importedInstances.size();

        sim_cfg.numUniqueScenes = load_result.uniqueSceneInfos.size();
        sim_cfg.uniqueScenes = (UniqueScene *)cu::allocGPU(
                sizeof(UniqueScene) * load_result.uniqueSceneInfos.size());

        sim_cfg.numWorlds = mgr_cfg.numWorlds;

        REQ_CUDA(cudaMemcpy(sim_cfg.importedInstances, 
                    load_result.importedInstances.data(),
                    sizeof(ImportedInstance) *
                    load_result.importedInstances.size(),
                    cudaMemcpyHostToDevice));

        REQ_CUDA(cudaMemcpy(sim_cfg.uniqueScenes, 
                    load_result.uniqueSceneInfos.data(),
                    sizeof(UniqueScene) *
                    load_result.uniqueSceneInfos.size(),
                    cudaMemcpyHostToDevice));


        if (render_mgr.has_value()) {
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

        uint32_t raycast_output_resolution = mgr_cfg.raycastOutputResolution;
        CudaBatchRenderConfig::RenderMode rt_render_mode;

        // If the rasterizer is enabled, disable the raycaster
        if (mgr_cfg.enableBatchRenderer) {
            raycast_output_resolution = 0;
        } else {
            rt_render_mode = CudaBatchRenderConfig::RenderMode::RGBD;
        }


        MWCudaExecutor gpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(Sim::WorldInit),
            .userConfigPtr = (void *)&sim_cfg,
            .numUserConfigBytes = sizeof(Sim::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = mgr_cfg.numWorlds,
            .numTaskGraphs = (uint32_t)TaskGraphID::NumTaskGraphs,
            .numExportedBuffers = (uint32_t)ExportID::NumExports, 
        }, {
            { RENDERER_SRC_LIST },
            { RENDERER_COMPILE_FLAGS },
            CompileConfig::OptMode::LTO,
        }, cu_ctx, 
        mgr_cfg.enableBatchRenderer ? Optional<madrona::CudaBatchRenderConfig>::none() : 
            madrona::CudaBatchRenderConfig {
                .renderMode = rt_render_mode,
                // .importedAssets = &imported_assets,
                .geoBVHData = render::AssetProcessor::makeBVHData(imported_assets.objects),
                .materialData = render::AssetProcessor::initMaterialData(
                        imported_assets.materials.data(), imported_assets.materials.size(),
                        imported_assets.textures.data(), imported_assets.textures.size()),
                .renderResolution = raycast_output_resolution,
                .nearPlane = 3.f,
                .farPlane = 1000.f
        });

        MWCudaLaunchGraph step_graph = gpu_exec.buildLaunchGraph(
                TaskGraphID::Step);
        MWCudaLaunchGraph render_setup_graph = gpu_exec.buildLaunchGraph(
                TaskGraphID::Render);

        Optional<MWCudaLaunchGraph> render_graph = [&]() -> Optional<MWCudaLaunchGraph> {
            if (mgr_cfg.enableBatchRenderer) {
                return Optional<MWCudaLaunchGraph>::none();
            } else {
                return gpu_exec.buildRenderGraph();
            }
        } ();

        Action *agent_actions_buffer = 
            (Action *)gpu_exec.getExported((uint32_t)ExportID::Action);

        return new CUDAImpl {
            mgr_cfg,
            agent_actions_buffer,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(gpu_exec),
            std::move(step_graph),
            std::move(render_setup_graph),
            std::move(render_graph)
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        FATAL("This environment doesn't support CPU backend");
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{
    // Currently, there is no way to populate the initial set of observations
    // without stepping the simulations in order to execute the taskgraph.
    // Therefore, after setup, we step all the simulations with a forced reset
    // that ensures the first real step will have valid observations at the
    // start of a fresh episode in order to compute actions.
    //
    // This will be improved in the future with support for multiple task
    // graphs, allowing a small task graph to be executed after initialization.
    const char *num_agents_str = getenv("HIDESEEK_NUM_AGENTS");
    if (num_agents_str) {
        uint32_t num_agents = std::stoi(num_agents_str);
        numAgents = num_agents;
    } else {
        numAgents = 1;
    }
    
    step();
}

Manager::~Manager() {}

void Manager::step()
{
    impl_->run();

    if (impl_->headlessMode) {
        if (impl_->cfg.enableBatchRenderer) {
            impl_->renderMgr->readECS();
        }
    } else {
        if (impl_->renderMgr.has_value()) {
            impl_->renderMgr->readECS();
        }
    }

    if (impl_->cfg.enableBatchRenderer) {
        impl_->renderMgr->batchRender();
    }
}

Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::Action, TensorElementType::Int32,
        {
            impl_->cfg.numWorlds,
            numAgents,
            4,
        });
}

Tensor Manager::rgbTensor() const
{
    const uint8_t *rgb_ptr = impl_->renderMgr->batchRendererRGBOut();

    return Tensor((void*)rgb_ptr, TensorElementType::UInt8, {
        impl_->cfg.numWorlds,
        numAgents,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);
}

Tensor Manager::depthTensor() const
{
    const float *depth_ptr = impl_->renderMgr->batchRendererDepthOut();

    return Tensor((void *)depth_ptr, TensorElementType::Float32, {
        impl_->cfg.numWorlds,
        numAgents,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

Tensor Manager::raycastTensor() const
{
    uint32_t pixels_per_view = impl_->raycastOutputResolution *
        impl_->raycastOutputResolution;
    return impl_->exportTensor(ExportID::Raycast,
                               TensorElementType::UInt8,
                               {
                                   impl_->cfg.numWorlds*numAgents,
                                   pixels_per_view * 4,
                               });
}

void Manager::setAction(int32_t world_idx,
                        int32_t agent_idx,
                        int32_t move_amount,
                        int32_t move_angle,
                        int32_t rotate,
                        int32_t grab,
                        int32_t x,
                        int32_t y,
                        int32_t z,
                        int32_t rot,
                        int32_t vrot)
{
    Action action { 
        .moveAmount = move_amount,
        .moveAngle = move_angle,
        .rotate = rotate,
        .grab = grab,
        .x = x,
        .y = y,
        .z = z,
        .rot = rot,
        .vrot = vrot,
    };

    auto *action_ptr = impl_->agentActionsBuffer +
        world_idx * numAgents + agent_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(action_ptr, &action, sizeof(Action),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

render::RenderManager & Manager::getRenderManager()
{
    return *impl_->renderMgr;
}

}
