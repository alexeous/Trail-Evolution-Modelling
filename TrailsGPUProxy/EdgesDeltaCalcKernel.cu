#include "EdgesDeltaCalcKernel.h"
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>

#ifndef __CUDACC__ // visual studio doesn't show functions declared in device_atomic_function.h without this
#define UNDEF__CUDACC__
#define __CUDACC__
#endif
#include <device_atomic_functions.h>
#include <math_functions.h>
#include <device_functions.h>
#ifdef UNDEF__CUDACC__
#undef UNDEF__CUDACC__
#undef __CUDACC__
#endif


namespace TrailEvolutionModelling {
	namespace GPUProxy {

		using FourFloats = thrust::tuple<const float&, const float&, const float&, const float&>;

		struct EdgesDeltaFunctor : public thrust::binary_function<FourFloats, FourFloats, float> {
			template <int N>
			__host__ __device__ inline float absdiff(const FourFloats& a, const FourFloats& b) const {
				// not using 'fabsf(x - y)' because 'x' and 'y' could be infinite 
				// (but that can only occur simultaneously when the edge is not walkable)
				return fdimf(a.get<N>(), b.get<N>()) + fdimf(b.get<N>(), a.get<N>());
			}
			
			__host__ __device__ float operator()(FourFloats last, FourFloats curr) {
				return absdiff<0>(curr, last) + absdiff<1>(curr, last) + absdiff<2>(curr, last) + absdiff<3>(curr, last);
				//constexpr float eps = 0.05f;
				//return (absdiff<0>(curr, last) > eps) +
				//	(absdiff<1>(curr, last) > eps) +
				//	(absdiff<2>(curr, last) > eps) +
				//	(absdiff<3>(curr, last) > eps);
				//return max(absdiff<0>(curr, last), max(absdiff<1>(curr, last), max(absdiff<2>(curr, last), absdiff<3>(curr, last))));
			}
		};


		float CalcEdgesDelta(EdgesWeightsDevice* lastWeights, EdgesWeightsDevice* currentWeights, int graphW, int graphH) 
		{
			using DP = thrust::device_ptr<float>;

			size_t arraySize = (graphW + 1) * (graphH + 1);
			auto lastWeightsStart = thrust::make_zip_iterator(thrust::make_tuple(
				DP(lastWeights->horizontal), DP(lastWeights->vertical),
				DP(lastWeights->leftDiagonal), DP(lastWeights->rightDiagonal)
			));
			auto lastWeightsEnd = thrust::make_zip_iterator(thrust::make_tuple(
				DP(lastWeights->horizontal + arraySize), DP(lastWeights->vertical + arraySize),
				DP(lastWeights->leftDiagonal + arraySize), DP(lastWeights->rightDiagonal + arraySize)
			));
			auto currWeightsStart = thrust::make_zip_iterator(thrust::make_tuple(
				DP(currentWeights->horizontal), DP(currentWeights->vertical),
				DP(currentWeights->leftDiagonal), DP(currentWeights->rightDiagonal)
			));

			return thrust::inner_product(lastWeightsStart, lastWeightsEnd, currWeightsStart, 0.0f, thrust::plus<float>(), EdgesDeltaFunctor());
		}

	}
}