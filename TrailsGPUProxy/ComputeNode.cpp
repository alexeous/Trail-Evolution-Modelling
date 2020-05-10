#include "ComputeNode.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		inline bool TrailEvolutionModelling::GPUProxy::ComputeNode::IsStart() {
			return dirNext_isStart >> 3;
		}

		inline void TrailEvolutionModelling::GPUProxy::ComputeNode::SetStart(bool start) {
			dirNext_isStart = (dirNext_isStart & 7) | (start << 3);
		}

		inline uint8_t TrailEvolutionModelling::GPUProxy::ComputeNode::GetDirNext() {
			return dirNext_isStart & 7;
		}

		inline void TrailEvolutionModelling::GPUProxy::ComputeNode::SetDirNext(uint8_t dirNext) {
			dirNext_isStart = ((dirNext_isStart >> 3) << 3) | dirNext;
		}

	}
}