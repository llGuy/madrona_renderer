#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>

#include "args.hpp"
#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"

#include <filesystem>
#include <fstream>
#include <imgui.h>

#include <stb_image_write.h>

#include <madrona/math.hpp>

using namespace madrona;
using namespace madrona::viz;
using namespace madrona::math;

void transposeImage(char *output, 
                    const char *input,
                    uint32_t res)
{
    for (uint32_t y = 0; y < res; ++y) {
        for (uint32_t x = 0; x < res; ++x) {
            output[4*(y + x * res) + 0] = input[4*(x + y * res) + 0];
            output[4*(y + x * res) + 1] = input[4*(x + y * res) + 1];
            output[4*(y + x * res) + 2] = input[4*(x + y * res) + 2];
            output[4*(y + x * res) + 3] = input[4*(x + y * res) + 3];
        }
    }
}

int main(int argc, char *argv[])
{
    using namespace madRender;

    run::ViewerRunArgs args = run::parseViewerArgs(argc, argv);

    WindowManager wm {};
    WindowHandle window = wm.makeWindow("Habitat Viewer", 
            args.windowWidth, args.windowHeight);

    render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });

    uint32_t output_resolution = args.batchRenderWidth;
    uint32_t num_worlds = args.numWorlds;



    Manager::Config::RenderConfig rcfg;
    { // Test this out
        rcfg.assetPaths = (const char **)malloc(2 * sizeof(const char *));
        rcfg.numAssetPaths = 2;

        rcfg.assetPaths[0] = "/home/luc/Development/madrona_renderer/data/wall_render.obj";
        rcfg.assetPaths[1] = "/home/luc/Development/madrona_renderer/data/plane.obj";

        rcfg.importedInstances = (ImportedInstance *)malloc(2 * sizeof(ImportedInstance));
        rcfg.numInstances = 2;

        rcfg.importedInstances[0].position = Vector3{ 0.f, 0.f, 15.f };
        rcfg.importedInstances[0].rotation = Quat::angleAxis(pi_d2, { 1.f, 0.f, 0.f });
        rcfg.importedInstances[0].scale = Diag3x3{ 10.f, 10.f, 10.f };
        rcfg.importedInstances[0].objectID = 0;

        rcfg.importedInstances[1].position = Vector3 { 0.f, 0.f, 0.f };
        rcfg.importedInstances[1].rotation = Quat::angleAxis(pi_d2, { 0.f, 0.f, 1.f });
        rcfg.importedInstances[1].scale = Diag3x3{ 0.01f, 0.01f, 0.01f };
        rcfg.importedInstances[1].objectID = 1;

        rcfg.cameras = (ImportedCamera *)malloc(1 * sizeof(ImportedCamera));
        rcfg.numCameras = 1;

        rcfg.cameras[0].position = { -30.f, -30.f, 15.f };
        rcfg.cameras[0].rotation = Quat::angleAxis(0.2f, { 1.f, 0.f, 1.f });

        rcfg.worlds = (Sim::WorldInit *)malloc(num_worlds * sizeof(Sim::WorldInit));

        for (int i = 0; i < num_worlds; ++i) {
            rcfg.worlds[i].numInstances = 2;
            rcfg.worlds[i].instancesOffset = 0;
            rcfg.worlds[i].numCameras = 1;
            rcfg.worlds[i].camerasOffset = 0;
        }
    }


    // Create the simulation manager
    Manager mgr({
        .gpuID = 0,
        .numWorlds = num_worlds,
        .renderMode = (Manager::RenderMode)args.renderMode,
        .batchRenderViewWidth = output_resolution,
        .batchRenderViewHeight = output_resolution,
        .extRenderAPI = wm.gpuAPIManager().backend(),
        .extRenderDev = render_gpu.device(),
        .headlessMode = false,
        .rcfg = rcfg,
    });
    float camera_move_speed = 10.f;

    math::Vector3 initial_camera_position = { 0, 0.f, 30 };

    math::Quat initial_camera_rotation =
        (math::Quat::angleAxis(-math::pi / 2.f, math::up) *
        math::Quat::angleAxis(-math::pi / 2.f, math::right)).normalize();


    // Create the viewer viewer
    viz::Viewer viewer(mgr.getRenderManager(), window.get(), {
        .numWorlds = num_worlds,
        .simTickRate = 120,
        .cameraMoveSpeed = camera_move_speed * 7.f,
        .cameraPosition = initial_camera_position,
        .cameraRotation = initial_camera_rotation,
    });

    // Main loop for the viewer viewer
    viewer.loop(
    [&mgr](CountT world_idx, const Viewer::UserInput &input)
    {
    },
    [&mgr](CountT world_idx, CountT agent_idx,
           const Viewer::UserInput &input)
    {
    }, [&]() {
        mgr.step();
    }, [&]() {
        {
            uint32_t num_image_x = 1;
            uint32_t num_image_y = 1;

            uint32_t num_images_total = num_image_x * num_image_y;

            unsigned char* print_ptr;
#ifdef MADRONA_CUDA_SUPPORT
            int64_t num_bytes = 4 * output_resolution * output_resolution * num_images_total;
            print_ptr = (unsigned char*)cu::allocReadback(num_bytes);
#else
            print_ptr = nullptr;
#endif

            char *raycast_tensor = (char *)(mgr.raycastTensor().devicePtr());

            uint32_t bytes_per_image = 4 * output_resolution * output_resolution;

            uint32_t image_idx = viewer.getCurrentWorldID() * 1 + 
                std::max(viewer.getCurrentViewID(), (CountT)0);

            uint32_t base_image_idx = num_images_total * (image_idx / num_images_total);

            raycast_tensor += image_idx * bytes_per_image;

#ifdef MADRONA_CUDA_SUPPORT
            cudaMemcpy(print_ptr, raycast_tensor,
                    num_bytes,
                    cudaMemcpyDeviceToHost);
            raycast_tensor = (char *)print_ptr;
#endif

            ImGui::Begin("Raycast");

            auto draw2 = ImGui::GetWindowDrawList();
            ImVec2 windowPos = ImGui::GetWindowPos();
            char *raycasters = raycast_tensor;

            int vertOff = 70;

            float pixScale = 3;
            int extentsX = (int)(pixScale * output_resolution);
            int extentsY = (int)(pixScale * output_resolution);

            for (int image_y = 0; image_y < num_image_y; ++image_y) {
                for (int image_x = 0; image_x < num_image_x; ++image_x) {
                    for (int i = 0; i < output_resolution; i++) {
                        for (int j = 0; j < output_resolution; j++) {
                            uint32_t linear_image_idx = image_y + image_x * num_image_x;

                            uint32_t linear_idx = 4 * 
                                (j + (i + linear_image_idx * output_resolution) * output_resolution);

                            auto realColor = IM_COL32(
                                    (uint8_t)raycasters[linear_idx + 0],
                                    (uint8_t)raycasters[linear_idx + 1],
                                    (uint8_t)raycasters[linear_idx + 2],
                                    255);

                            draw2->AddRectFilled(
                                    { ((i + image_y * output_resolution) * pixScale) + windowPos.x, 
                                    ((j + image_x * output_resolution) * pixScale) + windowPos.y + vertOff }, 
                                    { ((i + 1 + image_y * output_resolution) * pixScale) + windowPos.x,   
                                    ((j + image_x * output_resolution + 1) * pixScale) + +windowPos.y + vertOff },
                                    realColor, 0, 0);
                        }
                    }
                }
            }
            ImGui::End();
        }
    });
}
