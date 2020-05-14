#pragma once
#include <xhash>


namespace TrailEvolutionModelling {
	namespace GPUProxy {

        struct NodeIndex;

		struct NodeIndex {
            int i;
            int j;

            constexpr NodeIndex() : i(0), j(0) { }
            constexpr NodeIndex(int i, int j) : i(i), j(j) { }
            constexpr NodeIndex(const NodeIndex& other) : i(other.i), j(other.j) { }
            
            constexpr bool operator==(const NodeIndex& other) const { return i == other.i && j == other.j; }
            constexpr bool operator!=(const NodeIndex& other) const { return !(i == other.i && j == other.j); }
            constexpr NodeIndex operator+(const NodeIndex& other) const { return NodeIndex(i + other.i, j + other.j); }
            constexpr NodeIndex operator-(const NodeIndex& other) const { return NodeIndex(i - other.i, j - other.j); }
            float2 operator/(float c) const { return make_float2(i / c, j / c); }
            NodeIndex& operator+=(const NodeIndex& other) { 
                i += other.i;
                j += other.j;
                return *this;
            }
            NodeIndex& operator-=(const NodeIndex& other) { 
                i -= other.i;
                j -= other.j;
                return *this;
            }

            inline float SqrEuclideanDistance(const float2& other) {
                float di = i - (float)other.x;
                float dj = j - (float)other.y;
                return di * di + dj * dj;
            }

            static inline float SqrEuclideanDistance(const NodeIndex& a, const NodeIndex& b) {
                float di = (float)a.i - (float)b.i;
                float dj = (float)a.j - (float)b.j;
                return di * di + dj * dj;
            }

            inline float EuclideanDistance(const float2& other) {
                return sqrt(EuclideanDistance(other));
            }

            static inline float EuclideanDistance(const NodeIndex& a, const NodeIndex& b) {
                return (float)sqrt(SqrEuclideanDistance(a, b));
            }

		};

	}
}

namespace std {

    using namespace TrailEvolutionModelling::GPUProxy;

    template<>
    struct hash<NodeIndex> {
        std::size_t operator()(const NodeIndex& index) const {
            std::size_t result = 17;
            result = result * 31 + hash<int>()(index.i);
            result = result * 31 + hash<int>()(index.j);
            return result;
        }
    };

}