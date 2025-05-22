using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class GeometryFeedback
    {
        protected List<GeoAPI.Geometries.ICoordinate> _Coord = null;

        public GeometryFeedback()
        {
            CloseLoop = false;
            _Coord = new List<GeoAPI.Geometries.ICoordinate>();
            _Coord.Add(new USGS.Puma.NTS.Geometries.Coordinate());
            _Coord.Add(new USGS.Puma.NTS.Geometries.Coordinate(_Coord[0]));
        }

        public GeometryFeedback(GeoAPI.Geometries.ICoordinate startPoint)
        {
            CloseLoop = false;
            _Coord = new List<GeoAPI.Geometries.ICoordinate>();
            _Coord.Add(new USGS.Puma.NTS.Geometries.Coordinate());
            if (startPoint != null)
            {
                _Coord[0].X = startPoint.X;
                _Coord[0].Y = startPoint.Y;
            }
            _Coord.Add(new USGS.Puma.NTS.Geometries.Coordinate(_Coord[0]));

        }

        private bool _CloseLoop;
        public bool CloseLoop
        {
            get { return _CloseLoop; }
            set { _CloseLoop = value; }
        }

        public GeoAPI.Geometries.ICoordinate TrackPoint
        {
            get
            {
                if (_Coord.Count > 1)
                { return _Coord[_Coord.Count - 1]; }
                else
                { return null; }
            }

            set
            {
                if (value != null)
                {
                    if (_Coord.Count > 1)
                    {
                        _Coord[_Coord.Count - 1].X = value.X;
                        _Coord[_Coord.Count - 1].Y = value.Y;
                    }
                }
            }

        }

        public GeoAPI.Geometries.ICoordinate StartPoint
        {
            get
            {
                if (_Coord.Count > 0)
                { return _Coord[0]; }
                else
                { return null; }
            }

            set
            {
                if (value != null)
                {
                    if (_Coord.Count > 0)
                    {
                        _Coord[0].X = value.X;
                        _Coord[0].Y = value.Y;
                    }
                }
            }

        }

        public void AddPoint(GeoAPI.Geometries.ICoordinate point)
        {
            if (point != null)
            { _Coord.Add(new USGS.Puma.NTS.Geometries.Coordinate(point)); }
        }

        public virtual GeoAPI.Geometries.ICoordinate[] GetCoordinates()
        {
            int count = _Coord.Count;
            if (CloseLoop)
            { count += 1; }

            GeoAPI.Geometries.ICoordinate[] coords = new GeoAPI.Geometries.ICoordinate[count];
            for (int i = 0; i < _Coord.Count; i++)
            {
                coords[i] = new USGS.Puma.NTS.Geometries.Coordinate(_Coord[i]);
            }

            if (CloseLoop)
            { coords[coords.Length - 1] = new USGS.Puma.NTS.Geometries.Coordinate(coords[0]); }

            return coords;
        }

        public virtual GeoAPI.Geometries.IGeometry GetGeometry()
        {
            return null;
        }

    }
}
