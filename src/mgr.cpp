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

namespace madRender {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};


static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg)
{
    if (!mgr_cfg.headlessMode) {
        if (mgr_cfg.extRenderDev || 
                mgr_cfg.renderMode != Manager::RenderMode::Rasterizer) {
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
    const Optional<RenderGPUState> &render_gpu_state,
    uint32_t max_instances_per_world,
    uint32_t max_views_per_world)
{
    if (mgr_cfg.headlessMode && 
            mgr_cfg.renderMode != Manager::RenderMode::Rasterizer) {
        return Optional<render::RenderManager>::none();
    }

    if (!mgr_cfg.headlessMode) {
        if (!mgr_cfg.extRenderDev && 
                mgr_cfg.renderMode != Manager::RenderMode::Rasterizer) {
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
        .enableBatchRenderer =
            (mgr_cfg.renderMode == Manager::RenderMode::Rasterizer),
        .renderMode = render::RenderManager::Config::RenderMode::RGBD,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = max_views_per_world,
        .maxInstancesPerWorld = max_instances_per_world,
        .execMode = ExecMode::CUDA,
        .voxelCfg = {},
    });
}

struct Manager::Impl {
    Config cfg;

    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;

    uint32_t raycastOutputResolution;
    bool headlessMode;
    uint32_t totalNumCameras;
    uint32_t totalNumInstances;

    inline Impl(const Manager::Config &mgr_cfg,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr)
        : cfg(mgr_cfg),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr)),
          raycastOutputResolution(mgr_cfg.batchRenderViewWidth),
          headlessMode(mgr_cfg.headlessMode)
    {
        totalNumCameras = 0;
        totalNumInstances = 0;

        for (int i = 0; i < mgr_cfg.numWorlds; i++) {
            totalNumCameras += mgr_cfg.rcfg.worlds[i].numCameras;
            totalNumInstances += mgr_cfg.rcfg.worlds[i].numInstances;
        }
    }

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
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   MWCudaExecutor &&gpu_exec,
                   MWCudaLaunchGraph &&step_graph,
                   MWCudaLaunchGraph &&render_setup_graph,
                   Optional<MWCudaLaunchGraph> &&render_graph)
        : Impl(mgr_cfg,
               std::move(render_gpu_state), std::move(render_mgr)),
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

void importRawGeometry(
        const Manager::GeometryConfig *geo_cfg,
        madrona::imp::ImportedAssets &assets)
{
    using namespace madrona::imp;

    for (int i = 0; i < geo_cfg->numMeshes; ++i) {
        uint32_t vert_offset = geo_cfg->meshVertexOffsets[i];
        uint32_t index_offset = geo_cfg->meshIndexOffsets[i];
        int32_t material_idx = geo_cfg->meshMaterials[i];

        uint32_t vertex_count = (i < geo_cfg->numMeshes - 1) ?
            geo_cfg->meshVertexOffsets[i + 1] - vert_offset :
            geo_cfg->numVertices - vert_offset;
        uint32_t index_count = (i < geo_cfg->numMeshes - 1) ?
            geo_cfg->meshIndexOffsets[i + 1] - index_offset :
            geo_cfg->numIndices - index_offset;

        const Vector3 *src_vertices = geo_cfg->vertices + vert_offset;
        const Vector2 *src_uvs = geo_cfg->uvs + vert_offset;
        const uint32_t *src_indices = geo_cfg->indices + index_offset;

        DynArray<math::Vector3> new_vertices(vertex_count);
        new_vertices.resize(vertex_count, [](Vector3 *) {});
        memcpy(new_vertices.data(), src_vertices, sizeof(Vector3) * vertex_count);

        DynArray<math::Vector2> new_uvs(vertex_count);
        memcpy(new_uvs.data(), src_uvs, sizeof(Vector2) * vertex_count);

        DynArray<uint32_t> new_indices(index_count);
        memcpy(new_indices.data(), src_indices, sizeof(uint32_t) * index_count);

        DynArray<SourceMesh> mesh = {
            SourceMesh {
                .positions = new_vertices.data(),
                .normals = nullptr,
                .tangentAndSigns = nullptr,
                .uvs = new_uvs.data(),
                .indices = (uint32_t *)(geo_cfg->indices + index_offset),
                .faceCounts = nullptr,
                .faceMaterials = nullptr,
                .numVertices = vertex_count,
                .numFaces = index_count / 3,
                .materialIDX = (uint32_t)material_idx
            }
        };

        assets.geoData.meshArrays.push_back(std::move(mesh));

        assets.geoData.positionArrays.push_back(std::move(new_vertices));
        assets.geoData.uvArrays.push_back(std::move(new_uvs));
        assets.geoData.indexArrays.push_back(std::move(new_indices));

        assets.objects.push_back(SourceObject {
            .meshes = Span<SourceMesh>(
                    assets.geoData.meshArrays.back().data(), 1)
        });
    }
}

static imp::ImportedAssets loadRenderObjects(
        const Manager::GeometryConfig *geo_cfg,
        const char **paths,
        uint32_t num_paths,
        int32_t *mat_assignments,
        uint32_t num_mat_assignments,
        const AdditionalMaterial *additional_mats,
        uint32_t num_additional_mats,
        const char **additional_textures,
        uint32_t num_additional_textures,
        Optional<render::RenderManager> &render_mgr)
{
    using namespace madrona::imp;

#if 1
    std::vector<const char *> render_asset_cstrs;
    render_asset_cstrs.resize(num_paths);
    memcpy(render_asset_cstrs.data(), paths, 
           sizeof(const char *) * num_paths);

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
#endif


    // Load all the geometry manually
    importRawGeometry(geo_cfg, *render_assets);


    // Push the additional textures
    uint32_t old_tex_count = (uint32_t)render_assets->textures.size();
    for (int i = 0; i < num_additional_textures; ++i) {
        Optional<SourceTexture> tex = img_importer.importImage(
                additional_textures[i]);
        assert(tex.has_value());

        render_assets->textures.push_back(*tex);
    }

    uint32_t old_mat_count = 
        (uint32_t)render_assets->materials.size();

    // Push the additional materials
    for (int i = 0; i < num_additional_mats; ++i) {
        SourceMaterial mat = additional_mats[i];
        if (mat.textureIdx != -1) {
            // Adjust the ID of the texture
            mat.textureIdx += old_tex_count;
        }

        render_assets->materials.push_back(mat);
    }

#if 0
    // Now, do all the material assignments
    // TODO: Make sure to take into account multiple objects per asset_ file
    for (int i = 0; i < num_mat_assignments; ++i) {
        if (mat_assignments[i] != -1) {
            for (auto &mesh : render_assets->objects[i].meshes) {
                mesh.materialIDX = mat_assignments[i] + old_mat_count;
            }
        }
    }
#endif

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

#ifndef MADRONA_CUDA_SUPPORT
    static_assert(false, "Cuda support required");
#endif

    CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);

    Optional<RenderGPUState> render_gpu_state = initRenderGPUState(mgr_cfg);

    uint32_t max_instances_per_world = 1, max_views_per_world = 1;
    { // Calculate the two above
        for (int i = 0; i < mgr_cfg.numWorlds; ++i) {
            max_instances_per_world = std::max(
                    max_instances_per_world,
                    mgr_cfg.rcfg.worlds[i].numInstances);
            max_views_per_world = std::max(
                    max_views_per_world,
                    mgr_cfg.rcfg.worlds[i].numCameras);
        }
    }

    Optional<render::RenderManager> render_mgr = initRenderManager(
            mgr_cfg, render_gpu_state,
            max_instances_per_world,
            max_views_per_world);

    auto imported_assets = loadRenderObjects(
            &mgr_cfg.rcfg.geoCfg,
            mgr_cfg.rcfg.assetPaths,
            mgr_cfg.rcfg.numAssetPaths,
            mgr_cfg.rcfg.matAssignments,
            mgr_cfg.rcfg.numMatAssignments,
            mgr_cfg.rcfg.additionalMats,
            mgr_cfg.rcfg.numAdditionalMats,
            mgr_cfg.rcfg.additionalTextures,
            mgr_cfg.rcfg.numAdditionalTextures,
            render_mgr);

    // Allocate GPU memory for the instances
    sim_cfg.numImportedInstances = mgr_cfg.rcfg.numInstances;
    sim_cfg.importedInstances = (ImportedInstance *)cu::allocGPU(
            sizeof(ImportedInstance) * mgr_cfg.rcfg.numInstances);

    // Allocate GPU memory for the cameras
    sim_cfg.numImportedCameras = mgr_cfg.rcfg.numCameras;
    sim_cfg.importedCameras = (ImportedCamera *)cu::allocGPU(
            sizeof(ImportedCamera) * mgr_cfg.rcfg.numCameras);

    sim_cfg.numWorlds = mgr_cfg.numWorlds;

    // Copy the relevant stuff to the GPU.
    REQ_CUDA(cudaMemcpy(sim_cfg.importedInstances, 
                mgr_cfg.rcfg.importedInstances,
                sizeof(ImportedInstance) *
                    mgr_cfg.rcfg.numInstances,
                cudaMemcpyHostToDevice));

    REQ_CUDA(cudaMemcpy(sim_cfg.importedCameras, 
                mgr_cfg.rcfg.cameras,
                sizeof(ImportedCamera) *
                    mgr_cfg.rcfg.numCameras,
                cudaMemcpyHostToDevice));


    if (render_mgr.has_value()) {
        sim_cfg.renderBridge = render_mgr->bridge();
    } else {
        sim_cfg.renderBridge = nullptr;
    }

    HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);
    memcpy(world_inits.data(), mgr_cfg.rcfg.worlds,
           sizeof(Sim::WorldInit) * mgr_cfg.numWorlds);

    uint32_t raycast_output_resolution = mgr_cfg.batchRenderViewWidth;
    CudaBatchRenderConfig::RenderMode rt_render_mode;

    // If the rasterizer is enabled, disable the raycaster
    if (mgr_cfg.renderMode == Manager::RenderMode::Rasterizer) {
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
    (mgr_cfg.renderMode == Manager::RenderMode::Rasterizer) ?
        Optional<madrona::CudaBatchRenderConfig>::none() : 
        madrona::CudaBatchRenderConfig {
            .renderMode = rt_render_mode,
            .geoBVHData = render::AssetProcessor::makeBVHData(imported_assets.objects),
            .materialData = render::AssetProcessor::initMaterialData(
                    imported_assets.materials.data(), imported_assets.materials.size(),
                    imported_assets.textures.data(), imported_assets.textures.size()),
            .renderResolution = raycast_output_resolution,
            .nearPlane = 0.1f,
            .farPlane = 1000.f
    });

    MWCudaLaunchGraph step_graph = gpu_exec.buildLaunchGraph(
            TaskGraphID::Step);
    MWCudaLaunchGraph render_setup_graph = gpu_exec.buildLaunchGraph(
            TaskGraphID::Render);

    Optional<MWCudaLaunchGraph> render_graph = [&]() -> Optional<MWCudaLaunchGraph> {
        if (mgr_cfg.renderMode == Manager::RenderMode::Rasterizer) {
            return Optional<MWCudaLaunchGraph>::none();
        } else {
            return gpu_exec.buildRenderGraph();
        }
    } ();

    return new CUDAImpl {
        mgr_cfg,
        std::move(render_gpu_state),
        std::move(render_mgr),
        std::move(gpu_exec),
        std::move(step_graph),
        std::move(render_setup_graph),
        std::move(render_graph)
    };
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
        if (impl_->cfg.renderMode == Manager::RenderMode::Rasterizer) {
            impl_->renderMgr->readECS();
        }
    } else {
        if (impl_->renderMgr.has_value()) {
            impl_->renderMgr->readECS();
        }
    }

    if (impl_->cfg.renderMode == Manager::RenderMode::Rasterizer) {
        impl_->renderMgr->batchRender();
    }
}
Tensor Manager::rgbTensor() const
{
    if (impl_->cfg.renderMode == RenderMode::Rasterizer) {
        const uint8_t *rgb_ptr = impl_->renderMgr->batchRendererRGBOut();

        return Tensor((void*)rgb_ptr, TensorElementType::UInt8, {
            impl_->totalNumCameras,
            impl_->cfg.batchRenderViewHeight,
            impl_->cfg.batchRenderViewWidth,
            4,
        }, impl_->cfg.gpuID);
    } else {
        return impl_->exportTensor(ExportID::RaycastRGB,
                                   TensorElementType::UInt8,
                                   {
                                       impl_->totalNumCameras,
                                       impl_->raycastOutputResolution,
                                       impl_->raycastOutputResolution,
                                       4
                                   });
    }
}

Tensor Manager::depthTensor() const
{
    if (impl_->cfg.renderMode == RenderMode::Rasterizer) {
        const float *depth_ptr = impl_->renderMgr->batchRendererDepthOut();

        return Tensor((void*)depth_ptr, TensorElementType::Float32, {
            impl_->totalNumCameras,
            impl_->cfg.batchRenderViewHeight,
            impl_->cfg.batchRenderViewWidth,
            1,
        }, impl_->cfg.gpuID);
    } else {
        return impl_->exportTensor(ExportID::RaycastDepth,
                                   TensorElementType::Float32,
                                   {
                                       impl_->totalNumCameras,
                                       impl_->raycastOutputResolution,
                                       impl_->raycastOutputResolution
                                   });
    }
}

Tensor Manager::segmaskTensor() const
{
    if (impl_->cfg.renderMode == RenderMode::Rasterizer) {
        FATAL("Segmask not implemented for rasterizer");
    } else {
        return impl_->exportTensor(ExportID::RaycastSegmask,
                                   TensorElementType::Int32,
                                   {
                                       impl_->totalNumCameras,
                                       impl_->raycastOutputResolution,
                                       impl_->raycastOutputResolution
                                   });
    }
}

uint64_t Manager::rgbCudaPtr() const
{
    return (uint64_t)rgbTensor().devicePtr();
}

uint64_t Manager::depthCudaPtr() const
{
    return (uint64_t)depthTensor().devicePtr();
}

uint64_t Manager::segmaskCudaPtr() const
{
    return (uint64_t)segmaskTensor().devicePtr();
}

render::RenderManager & Manager::getRenderManager()
{
    return *impl_->renderMgr;
}

madrona::py::Tensor Manager::instancePositionTensor() const
{
    return impl_->exportTensor(ExportID::InstancePosition,
                               TensorElementType::Float32,
                               {
                                   impl_->totalNumInstances,
                                   3,
                               });
}

madrona::py::Tensor Manager::instanceRotationTensor() const
{
     return impl_->exportTensor(ExportID::InstanceRotation,
                               TensorElementType::Float32,
                               {
                                   impl_->totalNumInstances,
                                   4,
                               });
}

madrona::py::Tensor Manager::cameraPositionTensor() const
{
     return impl_->exportTensor(ExportID::CameraPosition,
                               TensorElementType::Float32,
                               {
                                   impl_->totalNumInstances,
                                   3,
                               });
}

madrona::py::Tensor Manager::cameraRotationTensor() const
{
     return impl_->exportTensor(ExportID::CameraRotation,
                               TensorElementType::Float32,
                               {
                                   impl_->totalNumInstances,
                                   4,
                               });
}

}
