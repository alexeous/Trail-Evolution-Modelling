#pragma once
#include <unordered_map>
#include <vector>
#include <xhash>
#include "Attractor.h"


namespace std {

    using Attractor = TrailEvolutionModelling::GPUProxy::Attractor;
    using RefAttractor = TrailEvolutionModelling::GraphTypes::Attractor;
    using Graph = TrailEvolutionModelling::GraphTypes::Graph;

    template<>
    struct hash<Attractor> {
        std::size_t operator()(const Attractor& attractor) const {
            std::size_t result = 17;
            result = result * 31 + hash<int>()(attractor.nodeI);
            result = result * 31 + hash<int>()(attractor.nodeJ);
            return result;
        }
    };
}

namespace TrailEvolutionModelling {
    namespace GPUProxy {

        using Attractor = TrailEvolutionModelling::GPUProxy::Attractor;
        using RefAttractor = TrailEvolutionModelling::GraphTypes::Attractor;
        using Graph = TrailEvolutionModelling::GraphTypes::Graph;

        class AttractorsMap : public std::unordered_map<Attractor, std::vector<Attractor>> {
        public:
            AttractorsMap(Graph^ graph, array<RefAttractor^>^ refAttractors);
            float GetSumReachablePerformance(const Attractor& attractor) const;

        public:
            static bool CanReach(Graph^ graph, RefAttractor^ a, RefAttractor^ b);
            static std::vector<Attractor> ConvertRefAttractors(array<RefAttractor^>^);

        private:
            std::unordered_map<Attractor, float> sumReachablePerformances;
        };

    }
}