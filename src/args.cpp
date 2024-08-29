#include "args.hpp"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <string>

namespace run {

ViewerRunArgs parseViewerArgs(int argc, char **argv)
{
    auto usage_err = [argv]() {
        fprintf(stderr, "%s [NUM_WORLDS] [rt|rast] [WINDOW_WIDTH] [WINDOW_HEIGHT] [BATCH_WIDTH] [BATCH_HEIGHT] [extra...]\n", argv[0]);
        exit(EXIT_FAILURE);
    };

    ViewerRunArgs run_args = {};

    if (argc < 7) {
        usage_err();
    }

    run_args.argCounter = 0;
    for (int i = 1; i < argc; ++i) {
        char *arg = argv[i];

        ++run_args.argCounter;

        if (i == 1) {
            run_args.numWorlds = std::stoi(arg);
        } else if (i == 2) {
            if (!strcmp("rt", arg)) {
                run_args.renderMode = RenderMode::Raycaster;
            } else if (!strcmp("rast", arg)) {
                run_args.renderMode = RenderMode::Rasterizer;
            } else {
                usage_err();
            }
        } else if (i == 3) {
            run_args.windowWidth = std::stoi(arg);
        } else if (i == 4) {
            run_args.windowHeight = std::stoi(arg);
        } else if (i == 5) {
            run_args.batchRenderWidth = std::stoi(arg);
        } else if (i == 6) {
            run_args.batchRenderHeight = std::stoi(arg);
        }
    }

    return run_args;
}

HeadlessRunArgs parseHeadlessArgs(int argc, char **argv)
{
    auto usage_err = [argv]() {
        fprintf(stderr, "%s [NUM_WORLDS] [NUM_STEPS] [rt|rast] [BATCH_WIDTH] [BATCH_HEIGHT] [--dump-last-frame file_name_without_extension]\n", argv[0]);
        exit(EXIT_FAILURE);
    };

    HeadlessRunArgs run_args = {};
    run_args.dumpOutputFile = false;

    if (argc != 6 && argc != 8) {
        usage_err();
    }

    run_args.argCounter = 0;
    for (int i = 1; i < argc; ++i) {
        char *arg = argv[i];
        ++run_args.argCounter;

        if (i == 1) {
            run_args.numWorlds = std::stoi(arg);
        } else if (i == 2) {
            run_args.numSteps = std::stoi(arg);
        } else if (i == 3) {
            if (!strcmp("rt", arg)) {
                run_args.renderMode = RenderMode::Raycaster;
            } else if (!strcmp("rast", arg)) {
                run_args.renderMode = RenderMode::Rasterizer;
            } else {
                usage_err();
            }
        } else if (i == 4) {
            run_args.batchRenderWidth = std::stoi(arg);
        } else if (i == 5) {
            run_args.batchRenderHeight = std::stoi(arg);
        } else if (i == 6) {
            if (!strcmp("--dump-last-frame", arg)) {
                run_args.dumpOutputFile = true;
            }
        } else if (i == 7) {
            std::string output_file = std::string(arg);
            run_args.outputFileName = output_file;
        }
    }

    return run_args;
}

}
