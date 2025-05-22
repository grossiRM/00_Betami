using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class LineStringFeedback : GeometryFeedback
    {
        public LineStringFeedback() : base()
        {
        }

        public LineStringFeedback(GeoAPI.Geometries.ICoordinate startingPoint) : base(startingPoint)
        {
        }

        public override GeoAPI.Geometries.IGeometry GetGeometry()
        {
            GeoAPI.Geometries.ICoordinate[] coords = GetCoordinates();
            USGS.Puma.NTS.Geometries.LineString geom = new USGS.Puma.NTS.Geometries.LineString(coords);
            return geom as GeoAPI.Geometries.IGeometry;
        }
    }
}
