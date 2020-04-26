using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Schema;
using System.Xml.Serialization;

namespace TrailEvolutionModelling.MapObjects
{
    [Serializable]
    struct AreaAttributes
    {
        public bool IsWalkable { get; set; }
        public float Weight { get; set; }
        public bool IsTramplable { get; set; }
    }
}
