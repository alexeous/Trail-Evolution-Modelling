#pragma once
#include <unordered_map>
#include <memory>
#include <atomic>
#include <gcroot.h>
#include "Attractor.h"
#include "AttractorsMap.h"
#include "ComputeNodesHost.h"
#include "PathReconstructor.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		class WavefrontCompletenessTable {
		private:
			enum Status { Unreachable, Blank, HalfCompleted, Completed };

			struct Cell {
				std::atomic<ComputeNodesHost*> rowResult;
				std::atomic<ComputeNodesHost*> colResult;
				std::atomic<Status> status;

				inline Status AdvanceStatus() {
					Status expected = Status::Blank;
					if(!status.compare_exchange_strong(expected, Status::HalfCompleted)) {
						if(expected == Status::Unreachable) {
							return Status::Unreachable;
						}
						// Already was HalfCompleted
						return (status = Status::Completed);
					}
					return Status::HalfCompleted;
				}
			};

		public:
			const int numPaths;

		public:
			WavefrontCompletenessTable(const AttractorsMap& attractors, PathReconstructor* pathReconstructor);
			~WavefrontCompletenessTable();

			void ResetCompleteness();
			void SetCompleted(const Attractor& attractor, ComputeNodesHost* calculatedNodes);

		private:
			void InitTable(const AttractorsMap& attractors);
			void InitAttractors(const AttractorsMap& attractors);
			int CountPaths(const AttractorsMap& attractors);
			inline Attractor GetRowAttractor(int row) const { return attractors[row]; }
			inline Attractor GetColumnAttractor(int col) const { return attractors[size - 1 - col]; }
			inline int GetRowIndex(const Attractor &attractor) const { return attractorToIndex.at(attractor); }
			inline int GetColumnIndex(const Attractor& attractor) const { return size - 1 - attractorToIndex.at(attractor); }

			inline Cell& GetCell(int row, int column)
				{ return table[column + row * size]; }

		private:
			int size;
			std::vector<Attractor> attractors;
			std::unordered_map<Attractor, int> attractorToIndex;
			Cell* table;

			PathReconstructor* pathReconstructor;
		};

	}
}