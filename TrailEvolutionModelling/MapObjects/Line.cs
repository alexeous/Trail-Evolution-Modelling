﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui.Geometries;
using Mapsui.Styles;
using TrailEvolutionModelling.Util;

namespace TrailEvolutionModelling.MapObjects
{
    public class Line : MapObject
    {
        public override IList<Point> Vertices => LineString.Vertices;
        public override bool AreVerticesLooped => false;
        public override string DisplayedName => AreaType?.DisplayedName ?? "<Line>";
        public override int MinimumVertices => 2;

        private LineString LineString => (LineString)Geometry;

        public override double Distance(Point p)
        {
            return LineString.Distance(p);
        }

        protected override IGeometry CreateGeometry()
        {
            return new LineString();
        }

        protected override void InitGeometryFromText(string geometryText)
        {
            if (string.IsNullOrWhiteSpace(geometryText))
            {
                throw new ArgumentException($"{nameof(geometryText)} is null or blank");
            }

            LineString lineString;
            try
            {
                var geometry = Mapsui.Geometries.Geometry.GeomFromText(geometryText);
                lineString = geometry as LineString;
                if (lineString == null)
                {
                    throw new ArgumentException($"Geometry text is invalid: {geometryText}", nameof(geometryText));
                }
            }
            catch (Exception ex)
            {
                throw new ArgumentException($"Geometry text is invalid: {geometryText}", nameof(geometryText), ex);
            }

            Geometry = lineString;
        }

        protected override string ConvertGeometryToText()
        {
            return Geometry.AsText();
        }

        protected override VectorStyle CreateHighlighedStyle()
        {
            return new VectorStyle
            {
                Line = new Pen
                {
                    Color = new Color(240, 20, 20),
                    Width = 7
                }
            };
        }

        public override bool IntersectsLine(Point start, Point end)
        {
            return GeometryIntersections.DoesLineIntersectPolyline(start, end, Vertices);
        }
    }
}
