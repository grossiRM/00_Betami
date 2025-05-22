using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class PolygonFeedback : GeometryFeedback
    {
        public PolygonFeedback()
            : base()
        { }

        public PolygonFeedback(GeoAPI.Geometries.ICoordinate startingPoint)
            : base(startingPoint)
        { }

        public override GeoAPI.Geometries.ICoordinate[] GetCoordinates()
        {
            GeoAPI.Geometries.ICoordinate[] coords = base.GetCoordinates();
            GeoAPI.Geometries.ICoordinate[] newCoords = new GeoAPI.Geometries.ICoordinate[coords.Length + 1];
            
            for (int i = 0; i < coords.Length; i++)
            {
                newCoords[i] = coords[i];
            }
            newCoords[newCoords.Length - 1] = new USGS.Puma.NTS.Geometries.Coordinate(coords[0]);

            return newCoords;
        }
        public override GeoAPI.Geometries.IGeometry GetGeometry()
        {
            GeoAPI.Geometries.ICoordinate[] coords = this.GetCoordinates();
            USGS.Puma.NTS.Geometries.LinearRing shell = new USGS.Puma.NTS.Geometries.LinearRing(coords);
            USGS.Puma.NTS.Geometries.Polygon geom = new USGS.Puma.NTS.Geometries.Polygon(shell);
            return geom as GeoAPI.Geometries.IGeometry;
        }
    }
}
