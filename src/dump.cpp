#include "dump.hpp"
#include <madrona/cuda_utils.hpp>

#include <stb_image_write.h>
#include <math.h>

namespace run {

static void transposeImage(char *output, 
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

static void copyRow(ColorType color_type,
                    char *output_row, const char *input_row,
                    uint32_t output_resolution)
{
    if (color_type == ColorType::RGB) {
        memcpy(output_row, input_row, 4 * output_resolution);
    } else {
        float normalization = 255.f;
        for (uint32_t pixel = 0; pixel < output_resolution; ++pixel) {
            float *depth = (float *)input_row + pixel;

            // Can rep
            uint8_t depth_u8 = (uint8_t)
                (255.0f * fminf(*depth / normalization, 1.f));
            output_row[pixel * 4 + 0] = depth_u8;
            output_row[pixel * 4 + 1] = depth_u8;
            output_row[pixel * 4 + 2] = depth_u8;
            output_row[pixel * 4 + 3] = 255;
        }
    }
}

void dumpTiledImage(const DumpInfo &info)
{
    using namespace madrona;

    uint32_t num_images_total = info.numImages;
    uint32_t output_resolution = info.imageResolution;

    unsigned char* print_ptr;
    int64_t num_bytes = 4 * output_resolution * output_resolution * num_images_total;
    print_ptr = (unsigned char*)cu::allocReadback(num_bytes);

    char *raycast_tensor = (char *)info.gpuTensor;

    // 4 is the size of each pixel regardless of RGB or depth
    uint32_t bytes_per_image = 4 * output_resolution * output_resolution;
    uint32_t row_stride_bytes = 4 * output_resolution;

    uint32_t image_idx = 0;

    uint32_t base_image_idx = num_images_total * (image_idx / num_images_total);

    raycast_tensor += image_idx * bytes_per_image;

    cudaMemcpy(print_ptr, raycast_tensor,
            num_bytes,
            cudaMemcpyDeviceToHost);
    raycast_tensor = (char *)print_ptr;

    // This will give the height
    float heightf = ceilf(sqrtf((float)num_images_total));
    float widthf = ceilf(num_images_total / heightf);

    uint32_t num_images_y = (uint32_t)heightf;
    uint32_t num_images_x = (uint32_t)widthf;

    char *tmp_image_memory = (char *)calloc(bytes_per_image, 1);
    char *image_memory = (char *)calloc(
            bytes_per_image * num_images_x * num_images_y, 1);

    uint32_t output_num_pixels_x = num_images_x * output_resolution;

    [&] () {
        for (uint32_t image_y = 0; image_y < num_images_y; ++image_y) {
            for (uint32_t image_x = 0; image_x < num_images_x; ++image_x) {
                uint32_t image_idx = image_x + image_y * num_images_x;
                if (image_idx >= num_images_total) {
                    return;
                }

                const char *input_image = raycast_tensor + image_idx * bytes_per_image;

                transposeImage(tmp_image_memory, input_image, output_resolution);

                for (uint32_t row_idx = 0; row_idx < output_resolution; ++row_idx) {
                    const char *input_row = tmp_image_memory + row_idx * row_stride_bytes;

                    uint32_t output_pixel_x = image_x * output_resolution;
                    uint32_t output_pixel_y = image_y * output_resolution + row_idx;

                    char *output_row = image_memory + 4 * (output_pixel_x + output_pixel_y * output_num_pixels_x);

                    // memcpy(output_row, input_row, 4 * output_resolution);
                    copyRow(info.colorType, output_row, input_row, output_resolution);
                }
            }
        }
    }();

    std::string file_name = info.outputPath + ".png";
    stbi_write_png(file_name.c_str(), output_resolution * num_images_x, num_images_y * output_resolution,
            4, image_memory, 4 * num_images_x * output_resolution);

    free(image_memory);
    free(tmp_image_memory);
}

}
