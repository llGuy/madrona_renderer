#include "mgr.hpp"
#include "args.hpp"
#include "dump.hpp"

#include <cstdio>
#include <chrono>
#include <string>
#include <filesystem>
#include <fstream>
#include <random>

#include <stb_image_write.h>
#include <madrona/window.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/render/render_mgr.hpp>

using namespace madrona;

[[maybe_unused]] static void saveWorldActions(
    const HeapArray<int32_t> &action_store,
    int32_t total_num_steps,
    int32_t world_idx)
{
    const int32_t *world_base = action_store.data() + world_idx * total_num_steps * 2 * 3;

    std::ofstream f("/tmp/actions", std::ios::binary);
    f.write((char *)world_base,
            sizeof(uint32_t) * total_num_steps * 2 * 3);
}

int main(int argc, char *argv[])
{
    using namespace madEscape;

    run::HeadlessRunArgs args = run::parseHeadlessArgs(argc, argv);

    std::string glb_path = argv[args.argCounter];

    ExecMode exec_mode = ExecMode::CUDA;
    uint64_t num_worlds = args.numWorlds;
    uint64_t num_steps = args.numSteps;

    bool enable_batch_renderer = 
        (args.renderMode == run::RenderMode::Rasterizer);

    uint32_t output_resolution = args.batchRenderWidth;

    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = (uint32_t)num_worlds,
        .autoReset = false,
        .enableBatchRenderer = enable_batch_renderer,
        .batchRenderViewWidth = output_resolution,
        .batchRenderViewHeight = output_resolution,
        .raycastOutputResolution = output_resolution,
        .headlessMode = true,
        .glbPath = glb_path
    });

    auto start = std::chrono::system_clock::now();

    for (CountT i = 0; i < (CountT)num_steps; i++) {
        mgr.step();
    }

    if (args.dumpOutputFile) {
        run::dumpTiledImage({
            .outputPath = args.outputFileName,
            .gpuTensor = (void *)mgr.raycastTensor().devicePtr(),
            .numImages = (uint32_t)num_worlds,
            .imageResolution = output_resolution
        });
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    float fps = (double)num_steps * (double)num_worlds / elapsed.count();
    printf("FPS %f\n", fps);
    printf("Average total step time: %f ms\n",
           1000.0f * elapsed.count() / (double)num_steps);
}
