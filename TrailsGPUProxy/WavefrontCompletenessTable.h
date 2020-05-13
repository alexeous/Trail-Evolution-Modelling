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
				std::atomic<ComputeNodesHost*> sourceResult;
				std::atomic<ComputeNodesHost*> drainResult;
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
			WavefrontCompletenessTable(const AttractorsMap& attractors, PathReconstructor* pathReconstructor);
			~WavefrontCompletenessTable();

			void ResetCompleteness();
			void SetCompleted(const Attractor& attractor, ComputeNodesHost* calculatedNodes);
			void WaitForAll();
			void CancelWait();

		private:
			void InitIndexMaps(const AttractorsMap& attractors);
			void InitTable(const AttractorsMap& attractors);
			int CountJobs(const AttractorsMap& attractors);
			int GetIndex(const Attractor& source, const Attractor& drain);
			void SetSourceCompleted(const Attractor& source, ComputeNodesHost* result);
			void SetDrainCompleted(const Attractor& drain, ComputeNodesHost* result);
			
			inline Cell& GetCell(const Attractor& source, const Attractor& drain) 
				{ return table[GetIndex(source, drain)]; }
			inline Cell& GetCell(int row, int column)
				{ return table[column + row * numColumns]; }

		private:
			int numRows;
			int numColumns;
			int numJobs;
			int pendingRemaining;
			std::unordered_map<Attractor, int> sourceToRow;
			std::unordered_map<Attractor, int> drainToColumn;
			Cell* table;

			PathReconstructor* pathReconstructor;
		};

	}
}