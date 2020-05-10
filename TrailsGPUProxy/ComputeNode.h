#pragma once
#include <cstdint>

namespace TrailEvolutionModelling {
    namespace GPUProxy {

        struct ComputeNode {
            float g;
            uint8_t dirNext_isStart;
            //int dirNext; // clockwise
            //int isStart;

            inline bool IsStart();
            inline void SetStart(bool start);
            inline uint8_t GetDirNext();
            inline void SetDirNext(uint8_t dirNext);
        };

    }
}