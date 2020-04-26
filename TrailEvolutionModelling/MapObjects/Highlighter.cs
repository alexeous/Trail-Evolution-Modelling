using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui.Providers;
using Mapsui.Styles;

namespace TrailEvolutionModelling.MapObjects
{
    class Highlighter
    {
        private Feature target;
        private bool isHighlighted;
        private IStyle highlightedStyle;

        public bool IsHighlighted
        {
            get => isHighlighted;
            set
            {
                if (value == isHighlighted) return;

                isHighlighted = value;
                if (isHighlighted)
                {
                    target.Styles.Add(highlightedStyle);
                }
                else
                {
                    target.Styles.Remove(highlightedStyle);
                }
            }
        }

        public Highlighter(Feature target, IStyle highlightedStyle)
        {
            if (target == null)
                throw new ArgumentNullException(nameof(target));
            if (highlightedStyle == null)
                throw new ArgumentNullException(nameof(highlightedStyle));
            
            this.target = target;
            this.highlightedStyle = highlightedStyle;
        }
    }
}
