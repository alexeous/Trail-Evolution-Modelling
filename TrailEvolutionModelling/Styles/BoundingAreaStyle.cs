using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui;
using Mapsui.Layers;
using Mapsui.Providers;
using Mapsui.Rendering;
using Mapsui.Rendering.Skia;
using Mapsui.Rendering.Skia.SkiaStyles;
using Mapsui.Styles;
using SkiaSharp;
using MapsuiPolygon = Mapsui.Geometries.Polygon;

namespace TrailEvolutionModelling.Styles
{
    class BoundingAreaStyle : VectorStyle
    {
        public BoundingAreaStyle()
        {
            Fill = new Brush(Color.FromArgb(140, 255, 255, 255));
            Outline = new Pen
            {
                Color = Color.Green,
                Width = 2
            };
        }
    }

    class BoundingAreaRenderer : ISkiaStyleRenderer
    {
        public bool Draw(SKCanvas canvas, IReadOnlyViewport viewport, ILayer layer, IFeature feature, IStyle style, ISymbolCache symbolCache)
        {
            if (!(feature.Geometry is MapsuiPolygon polygon && style is VectorStyle vectorStyle))
                return false;

            using (var path = polygon.ToSkiaPath(viewport, canvas.LocalClipBounds, 0))
            using (var paint = new SKPaint { IsAntialias = true })
            {
                paint.Style = SKPaintStyle.Stroke;
                paint.Color = vectorStyle.Outline.Color.ToSkia();
                paint.StrokeWidth = (float)vectorStyle.Outline.Width;
                canvas.DrawPath(path, paint);

                paint.Style = SKPaintStyle.Fill;
                paint.Color = vectorStyle.Fill.Color.ToSkia();
                path.FillType = SKPathFillType.InverseWinding;
                canvas.DrawPath(path, paint);
            }
            return true;
        }
    }
}
