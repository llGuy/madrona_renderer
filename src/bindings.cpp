#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>

using namespace madrona;
using namespace madrona::math;

namespace nb = nanobind;

namespace madRender {

NB_MODULE(madrona_renderer, m) {
    madrona::py::setupMadronaSubmodule(m);

    nb::enum_<Manager::RenderMode>(m, "RenderMode")
        .value("Rasterizer", Manager::RenderMode::Rasterizer)
        .value("Raytracer", Manager::RenderMode::Raytracer)
    ;

    nb::class_<ImportedInstance>(m, "ImportedAsset")
        .def("__init__", [](ImportedAsset *self,
                            std::string path,
                            std::optional<int64_t> mat_id) {
            new (self) ImportedAsset {
                .path = path,
                .matID = mat_id.has_value() ? (int32_t)mat_id.value() : -1,
            };
        }, nb::arg("path"),
           nb::arg("mat_id") = nb::none())
    ;

    nb::class_<ImportedInstance>(m, "AdditionalMaterial")
        .def("__init__", [](ImportedAsset *self,
                            const std::array<float, 4> &color,
                            std::optional<int64_t> texture_id,
                            float roughness,
                            float metalness) {
            new (self) AdditionalMaterial {
                .color = Vector4{ color[0], color[1], color[2], color[3] },
                .textureIdx = texture_id.has_value() ? 
                (int32_t)texture_id.value() : -1,
                .roughness = roughness,
                .metalness = metalness,
            };
        }, nb::arg("color"),
           nb::arg("texture_id") = nb::none(),
           nb::arg("roughness"),
           nb::arg("metalness"))
    ;

    nb::class_<ImportedInstance>(m, "ImportedInstance")
        .def("__init__", [](ImportedInstance *self,
                            const std::array<float, 3> &pos,
                            const std::array<float, 4> &rot,
                            const std::array<float, 3> &scale,
                            int64_t object_id) {
            new (self) ImportedInstance {
                .position = { pos[0], pos[1], pos[2] },
                .rotation = { rot[0], rot[1], rot[2], rot[3] },
                .scale = { scale[0], scale[1], scale[2] },
                .objectID = (int32_t)object_id
            };
        }, nb::arg("position"),
           nb::arg("rotation"),
           nb::arg("scale"),
           nb::arg("object_id"))
    ;

    nb::class_<ImportedCamera>(m, "ImportedCamera")
        .def("__init__", [](ImportedCamera *self,
                            const std::array<float, 3> &pos,
                            const std::array<float, 4> &rot) {
            new (self) ImportedCamera {
                .position = { pos[0], pos[1], pos[2] },
                .rotation = { rot[0], rot[1], rot[2], rot[3] }
            };
        }, nb::arg("position"),
           nb::arg("rotation"))
    ;

    nb::class_<Sim::WorldInit>(m, "WorldInit")
        .def("__init__", [](Sim::WorldInit *self,
                            int64_t num_instances,
                            int64_t instance_offset,
                            int64_t num_cameras,
                            int64_t camera_offset) {
            new (self) Sim::WorldInit {
                .numInstances = (uint32_t)num_instances,
                .instancesOffset = (uint32_t)instance_offset,
                .numCameras = (uint32_t)num_cameras,
                .camerasOffset = (uint32_t)camera_offset
            };
        }, nb::arg("num_instances"),
           nb::arg("instance_offset"),
           nb::arg("num_cameras"),
           nb::arg("camera_offset"))
    ;

#if 1
    nb::class_<Manager>(m, "MadronaRenderer")
        .def("__init__", [](Manager *self,
                            int gpu_id,
                            int num_worlds,
                            Manager::RenderMode render_mode,
                            int batch_render_view_width,
                            int batch_render_view_height,
                            const std::vector<ImportedAsset> &asset_paths,
                            const std::vector<AdditionalMaterial> &mats,
                            const std::vector<std::string> &texture_paths,
                            const std::vector<ImportedInstance> &instances,
                            const std::vector<ImportedCamera> &cameras,
                            const std::vector<Sim::WorldInit> &worlds) {
            std::vector<const char *> cstrs;
            std::vector<int32_t> mat_assignments;

            cstrs.resize(asset_paths.size());
            mat_assignments.resize(asset_paths.size());

            for (uint32_t i = 0; i < (uint32_t)asset_paths.size(); ++i) {
                cstrs[i] = asset_paths[i].path.c_str();
                mat_assignments[i] = asset_paths[i].matID;
            }

            std::vector<const char *> texture_cstrs;

            for (uint32_t i = 0; i < (uint32_t)texture_paths.size(); ++i) {
                texture_cstrs[i] = texture_paths[i].c_str();
            }

            new (self) Manager(Manager::Config {
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .renderMode = render_mode,
                .batchRenderViewWidth = (uint32_t)batch_render_view_width,
                .batchRenderViewHeight = (uint32_t)batch_render_view_height,
                .rcfg = {
                    .assetPaths = cstrs.data(),
                    .numAssetPaths = (uint32_t)cstrs.size(),
                    .matAssignments = mat_assignments.data(),
                    .numMatAssignments = (uint32_t)mat_assignments.size(),
                    .additionalMats = mats.data(),
                    .numAdditionalMats = (uint32_t)mats.size(),
                    .additionalTextures = texture_cstrs.data(),
                    .numAdditionalTextures = (uint32_t)texture_paths.size(),
                    .importedInstances = (ImportedInstance *)instances.data(),
                    .numInstances = (uint32_t)instances.size(),
                    .cameras = (ImportedCamera *)cameras.data(),
                    .numCameras = (uint32_t)cameras.size(),
                    .worlds = (Sim::WorldInit *)worlds.data(),
                },
            });
        }, nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("render_mode"),
           nb::arg("batch_render_view_width"),
           nb::arg("batch_render_view_height"),
           nb::arg("asset_paths"),
           nb::arg("materials"),
           nb::arg("texture_paths"),
           nb::arg("instances"),
           nb::arg("cameras"),
           nb::arg("worlds"))
        .def("step", &Manager::step)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("depth_tensor", &Manager::depthTensor)
        .def("segmask_tensor", &Manager::segmaskTensor)
        .def("rgb_cuda_ptr", &Manager::rgbCudaPtr)
        .def("depth_cuda_ptr", &Manager::depthCudaPtr)
        .def("segmask_cuda_ptr", &Manager::segmaskCudaPtr)
        .def("instance_position_tensor", &Manager::instancePositionTensor)
        .def("instance_rotation_tensor", &Manager::instanceRotationTensor)
        .def("camera_position_tensor", &Manager::cameraPositionTensor)
        .def("camera_rotation_tensor", &Manager::cameraRotationTensor)
    ;
#endif
}

}
