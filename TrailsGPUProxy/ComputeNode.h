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
            inline bool IsStart();
            inline void SetStart(bool start);
            inline uint8_t GetDirNext();
            inline void SetDirNext(uint8_t dirNext);
        };

    }
}