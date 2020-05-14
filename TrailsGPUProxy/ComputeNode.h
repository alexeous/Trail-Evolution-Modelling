#pragma once
#include <cstdint>
#include "NodeIndex.h"

namespace TrailEvolutionModelling {
    namespace GPUProxy {

        struct ComputeNode {
        public:
            float g;
        private:
            uint8_t dirNext_isStart;

        public:
            inline __host__ __device__ bool IsStart() const { return dirNext_isStart >> 3; }
            inline __host__ __device__ void SetStart(bool start) { dirNext_isStart = (dirNext_isStart & 7) | (start << 3); }
            inline __host__ __device__ uint8_t GetDirNext() const { return dirNext_isStart & 7; }
            inline __host__ __device__ void SetDirNext(uint8_t dirNext) { dirNext_isStart = ((dirNext_isStart >> 3) << 3) | dirNext; }

#ifndef __CUDACC__
            inline constexpr NodeIndex NextIndex(NodeIndex index) const {
                return index + DirectionToShift(GetDirNext());
            }
#endif

            static constexpr NodeIndex DirectionToShift(int direction) {
                constexpr NodeIndex shifts[] = {
                    {-1, -1}, {0, -1}, {1, -1},
                    {-1,  0},          {1,  0},
                    {-1,  1}, {0,  1}, {1,  1}
                };
                return shifts[direction];
            }
        };

    }
}