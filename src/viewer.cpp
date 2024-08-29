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

using namespace madrona;
using namespace madrona::viz;

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
    using namespace madEscape;

    run::ViewerRunArgs args = run::parseViewerArgs(argc, argv);

    std::string glb_path = argv[args.argCounter];

    // "Batch renderer" refers to the rasterizer.
    bool enable_batch_renderer = (args.renderMode == run::RenderMode::Rasterizer);

    WindowManager wm {};
    WindowHandle window = wm.makeWindow("Habitat Viewer", 
            args.windowWidth, args.windowHeight);

    render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });

    uint32_t output_resolution = args.batchRenderWidth;
    uint32_t num_worlds = args.numWorlds;

    // Create the simulation manager
    Manager mgr({
        .execMode = madrona::ExecMode::CUDA,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .randSeed = 5,
        .autoReset = false,
        .enableBatchRenderer = enable_batch_renderer,
        .batchRenderViewWidth = output_resolution,
        .batchRenderViewHeight = output_resolution,
        .extRenderAPI = wm.gpuAPIManager().backend(),
        .extRenderDev = render_gpu.device(),
        .raycastOutputResolution = output_resolution,
        .headlessMode = false,
        .glbPath = glb_path
    });
    float camera_move_speed = 10.f;

    math::Vector3 initial_camera_position = { 0, consts::worldLength / 2.f, 30 };

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
        using Key = Viewer::KeyboardKey;

        int32_t x = 0;
        int32_t y = 0;
        int32_t r = 2;
        int32_t g = 0;

        bool shift_pressed = input.keyPressed(Key::Shift);

        if (input.keyPressed(Key::W)) {
            y += 1;
        }
        if (input.keyPressed(Key::S)) {
            y -= 1;
        }

        if (input.keyPressed(Key::D)) {
            x += 1;
        }
        if (input.keyPressed(Key::A)) {
            x -= 1;
        }

        if (input.keyPressed(Key::Q)) {
            r += shift_pressed ? 2 : 1;
        }
        if (input.keyPressed(Key::E)) {
            r -= shift_pressed ? 2 : 1;
        }

        if (input.keyHit(Key::G)) {
            g = 1;
        }

        int32_t move_amount;
        if (x == 0 && y == 0) {
            move_amount = 0;
        } else if (shift_pressed) {
            move_amount = consts::numMoveAmountBuckets - 1;
        } else {
            move_amount = 1;
        }

        int32_t move_angle;
        if (x == 0 && y == 1) {
            move_angle = 0;
        } else if (x == 1 && y == 1) {
            move_angle = 1;
        } else if (x == 1 && y == 0) {
            move_angle = 2;
        } else if (x == 1 && y == -1) {
            move_angle = 3;
        } else if (x == 0 && y == -1) {
            move_angle = 4;
        } else if (x == -1 && y == -1) {
            move_angle = 5;
        } else if (x == -1 && y == 0) {
            move_angle = 6;
        } else if (x == -1 && y == 1) {
            move_angle = 7;
        } else {
            move_angle = 0;
        }

        x = 1;
        if (input.keyPressed(Key::W)) {
            x = 2;
        }
        if (input.keyPressed(Key::S)) {
            x = 0;
        }

        y = 1;
        if (input.keyPressed(Key::D)) {
            y = 2;
        }
        if (input.keyPressed(Key::A)) {
            y = 0;
        }

        int rot=1;
        if (input.keyPressed(Key::Q)) {
            rot = 2;
        }
        if (input.keyPressed(Key::E)) {
            rot = 0;
        }

        int vrot = 1;
        if (input.keyPressed(Key::T)) {
            vrot = 2;
        }
        if (input.keyPressed(Key::F)) {
            vrot = 0;
        }

        int z = 1;
        if (input.keyPressed(Key::O)) {
            z = 2;
        }

        if (input.keyPressed(Key::L)) {
            z = 0;
        }

        mgr.setAction(world_idx, agent_idx, move_amount, move_angle, 
                r, g, x, y, z, rot, vrot);
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
