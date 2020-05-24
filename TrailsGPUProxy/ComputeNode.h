#pragma once
#include <cstdint>
#include "NodeIndex.h"
#include <string>

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

            static inline void ThrowInvalidShift(const NodeIndex& shift) {
                std::string msg = "Invalid shift: (" + std::to_string(shift.i) + "; " + std::to_string(shift.j) + ")";
                throw std::exception(msg.c_str());
            }

            static inline int ShiftToDirection(const NodeIndex& shift) {
                switch(shift.j) {
                    case -1:
                        switch(shift.i) {
                            case -1: return 0;
                            case 0: return 1;
                            case 1: return 2;
                            default: ThrowInvalidShift(shift);
                        }
                    case 0:
                        switch(shift.i) {
                            case -1: return 3;
                            case 1: return 4;
                            default: ThrowInvalidShift(shift);
                        }
                    case 1:
                        switch(shift.i) {
                            case -1: return 5;
                            case 0: return 6;
                            case 1: return 7;
                            default: ThrowInvalidShift(shift);
                        }
                } 
                ThrowInvalidShift(shift);
            }
        };

    }
}