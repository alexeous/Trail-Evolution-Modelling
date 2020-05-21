#include "WavefrontCompletenessTable.h"
#include <atomic>

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		WavefrontCompletenessTable::WavefrontCompletenessTable(const AttractorsMap& attractors,
			PathReconstructor* pathReconstructor)
			: size((int)attractors.size()),
			  table(nullptr),
			  pathReconstructor(pathReconstructor),
			  numPaths(CountPaths(attractors))
		{
			InitAttractors(attractors);
			InitTable(attractors);
		}

		void WavefrontCompletenessTable::InitTable(const AttractorsMap& attractorsMap) {
			table = new Cell[size * size];
			for(int i = 0; i < size * size; i++) {
				Cell& cell = table[i];
				cell.status = Status::Unreachable;
				cell.rowResult = nullptr;
				cell.colResult = nullptr;
			}
			
			for(int row = 0; row < attractors.size(); row++) {
				Attractor rowAttr = GetRowAttractor(row);
				std::vector<Attractor> reachableAttrs = attractorsMap.at(rowAttr);
				for(int col = 0; col < size - row - 1; col++) {
					Attractor colAttr = GetColumnAttractor(col);
					bool reachable = std::find(reachableAttrs.begin(), reachableAttrs.end(), 
						colAttr) != reachableAttrs.end();
					if(reachable)
						GetCell(row, col).status = Status::Blank;
				}
			}
		}

		void WavefrontCompletenessTable::InitAttractors(const AttractorsMap& attrs) {
			int i = 0;
			for(auto pair : attrs) {
				attractors.push_back(pair.first);
				attractorToIndex[pair.first] = i;
				i++;
			}
		}

		int WavefrontCompletenessTable::CountPaths(const AttractorsMap& attractors) {
			int count = 0;
			for(auto pair : attractors) {
				count += (int)pair.second.size();
			}
			return count / 2; // div by 2 because it includes two-way paths
		}

		void WavefrontCompletenessTable::ResetCompleteness() {
			int tableSize = size * size;
			for(int i = 0; i < tableSize; i++) {
				Cell& cell = table[i];
				if(cell.status != Status::Unreachable) {
					cell.status = Status::Blank;
					cell.rowResult = nullptr;
					cell.colResult = nullptr;
				}
			}
		}

		void WavefrontCompletenessTable::SetCompleted(const Attractor& attractor, 
			ComputeNodesHost* calculatedNodes) 
		{
			int attrRow = GetRowIndex(attractor);
			int attrCol = GetColumnIndex(attractor);
			for(int col = 0; col < attrCol; col++) {
				Cell& cell = GetCell(attrRow, col);
				if(cell.status == Status::Unreachable)
					continue;

				cell.rowResult = calculatedNodes;
				if(cell.AdvanceStatus() == Status::Completed) {
					Attractor other = GetColumnAttractor(col);
					pathReconstructor->StartPathReconstruction(attractor, other, calculatedNodes, cell.colResult);
				}
			}
			for(int row = 0; row < attrRow; row++) {
				Cell& cell = GetCell(row, attrCol);
				if(cell.status == Status::Unreachable)
					continue;

				cell.colResult = calculatedNodes;
				if(cell.AdvanceStatus() == Status::Completed) {
					Attractor other = GetRowAttractor(row);
					pathReconstructor->StartPathReconstruction(attractor, other, calculatedNodes, cell.rowResult);
				}
			}
		}

		WavefrontCompletenessTable::~WavefrontCompletenessTable() {
			delete[] table;
		}

	}
}