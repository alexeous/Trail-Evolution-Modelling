#pragma once
#include <cstdint>

namespace TrailEvolutionModelling {
    namespace GPUProxy {

        struct ComputeNode {
        public:
            float g;
        private:
            uint8_t dirNext_isStart;

        public:
            inline bool IsStart() { return dirNext_isStart >> 3; }
            inline void SetStart(bool start) { dirNext_isStart = (dirNext_isStart & 7) | (start << 3); }
            inline uint8_t GetDirNext() { return dirNext_isStart & 7; }
            inline void SetDirNext(uint8_t dirNext) { dirNext_isStart = ((dirNext_isStart >> 3) << 3) | dirNext; }
        };

    }
}