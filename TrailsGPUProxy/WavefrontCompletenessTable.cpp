#include "WavefrontCompletenessTable.h"
#include <atomic>

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		WavefrontCompletenessTable::WavefrontCompletenessTable(const AttractorsMap& attractors,
			PathReconstructor* pathReconstructor)
			: numRows(attractors.GetSourceNumber()), 
			  numColumns(attractors.GetDrainNumber()),
			  table(nullptr),
			  pathReconstructor(pathReconstructor),
			  numJobs(CountJobs(attractors))
		{
			InitIndexMaps(attractors);
			InitTable(attractors);
		}

		void WavefrontCompletenessTable::InitIndexMaps(const AttractorsMap& attractors) {
			int row = 0;
			int column = 0;
			for(auto pair : attractors) {
				Attractor attractor = pair.first;
				if(attractor.isSource) {
					sourceToRow[attractor] = row;
					row++;
				}
				if(attractor.isDrain) {
					drainToColumn[attractor] = column;
					column++;
				}
			}
		}

		void WavefrontCompletenessTable::InitTable(const AttractorsMap& attractors) {
			int tableSize = numRows * numColumns;
			table = new Cell[tableSize];
			for(int i = 0; i < tableSize; i++) {
				Cell& cell = table[i];
				cell.status = Status::Unreachable;
				cell.sourceResult = nullptr;
				cell.drainResult = nullptr;
			}

			for(auto pair : attractors) {
				// it's enough to iterate over sources only to
				// process all source-drain pairs
				if(pair.first.isDrain)
					continue;
				
				Attractor source = pair.first;
				const std::vector<Attractor>& drains = pair.second;
				for(Attractor drain : drains) {
					GetCell(source, drain).status = Status::Blank;
				}
			}
		}

		int WavefrontCompletenessTable::CountJobs(const AttractorsMap& attractors) {
			return attractors.uniqueAttractors.size();
		}

		void WavefrontCompletenessTable::ResetCompleteness() {
			int tableSize = numRows * numColumns;
			for(int i = 0; i < tableSize; i++) {
				Cell& cell = table[i];
				if(cell.status != Status::Unreachable) {
					cell.status = Status::Blank;
					cell.sourceResult = nullptr;
					cell.drainResult = nullptr;
				}
			}
			pendingRemaining = numJobs;
		}

		void WavefrontCompletenessTable::SetCompleted(const Attractor& attractor, 
			ComputeNodesHost* calculatedNodes) 
		{
			if(attractor.isSource) {
				SetSourceCompleted(attractor, calculatedNodes);
			}
			if(attractor.isDrain) {
				SetDrainCompleted(attractor, calculatedNodes);
			}
		}

		void WavefrontCompletenessTable::WaitForAll() {
			while(pendingRemaining > 0) {
				_sleep(5);
			}
		}

		void WavefrontCompletenessTable::CancelWait() {
			pendingRemaining = 0;
		}

		void WavefrontCompletenessTable::SetSourceCompleted(const Attractor& source, 
			ComputeNodesHost* result)
		{
			int row = sourceToRow[source];
			for(auto pair : drainToColumn) {
				Cell& cell = GetCell(row, pair.second);
				if(cell.status == Status::Unreachable)
					continue;

				cell.sourceResult = result;
				if(cell.AdvanceStatus() == Status::Completed) {
					pathReconstructor->StartPathReconstruction(source, pair.first, result, cell.drainResult);
				}
			}
		}

		void WavefrontCompletenessTable::SetDrainCompleted(const Attractor& drain, 
			ComputeNodesHost* result)
		{
			int column = drainToColumn[drain];
			for(auto pair : sourceToRow) {
				Cell& cell = GetCell(pair.second, column);
				if(cell.status == Status::Unreachable)
					continue;

				cell.drainResult = result;
				if(cell.AdvanceStatus() == Status::Completed) {
					pathReconstructor->StartPathReconstruction(pair.first, drain, cell.sourceResult, result);
				}
			}
		}

		int WavefrontCompletenessTable::GetIndex(const Attractor& source, const Attractor& drain) {
			int row = sourceToRow[source];
			int col = drainToColumn[drain];
			return col + row * numColumns;
		}

		WavefrontCompletenessTable::~WavefrontCompletenessTable() {
			delete[] table;
		}

	}
}