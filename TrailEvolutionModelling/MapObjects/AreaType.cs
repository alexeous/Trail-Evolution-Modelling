using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Schema;
using System.Xml.Serialization;
using Mapsui.Styles;
using TrailEvolutionModelling.Styles;

namespace TrailEvolutionModelling.MapObjects
{
    [Serializable]
    class AreaType
    {
        public AreaAttributes Attributes { get; private set; }
        public AreaStyle Style { get; private set; }
        public string Name { get; private set; }
        public string DisplayedName { get; private set; }

        private AreaType() { }
    }
}
