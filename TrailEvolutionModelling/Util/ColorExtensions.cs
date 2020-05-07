using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui.Styles;

namespace TrailEvolutionModelling.Util
{
    static class ColorExtensions
    {
        public static Color Scale(this Color color, float factor, bool preserveOriginalAlpha = true)
        {
            int alpha = preserveOriginalAlpha ? color.A : (int)(color.A * factor);
            return Color.FromArgb(alpha, (int)(color.R * factor),
                                         (int)(color.G * factor),
                                         (int)(color.B * factor));
        }

        public static Color WithAlpha(this Color color, int alpha)
        {
            return Color.FromArgb(alpha, color.R, color.G, color.B);
        }
    }
}
