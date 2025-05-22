using System;
using System.Collections;
using System.Text;

using GeoAPI.Geometries;

using USGS.Puma.NTS.Geometries;

namespace USGS.Puma.NTS.Algorithm
{
    /// <summary>
    /// Computes the centroid of a point point.
    /// Algorithm:
    /// Compute the average of all points.
    /// </summary>
    /// <remarks></remarks>
    public class CentroidPoint
    {
        /// <summary>
        /// 
        /// </summary>
        private int ptCount = 0;
        /// <summary>
        /// 
        /// </summary>
        private ICoordinate centSum = new Coordinate();

        /// <summary>
        /// Initializes a new instance of the <see cref="T:System.Object"/> class.
        /// </summary>
        /// <remarks></remarks>
        public CentroidPoint() { }

        /// <summary>
        /// Adds the point(s) defined by a Geometry to the centroid total.
        /// If the point is not of dimension 0 it does not contribute to the centroid.
        /// </summary>
        /// <param name="geom">The point to add.</param>
        /// <remarks></remarks>
        public void Add(IGeometry geom)
        {
            if (geom is IPoint)             
                Add(geom.Coordinate);

            else if(geom is IGeometryCollection) 
            {
                IGeometryCollection gc = (IGeometryCollection) geom;
                foreach (IGeometry geometry in gc.Geometries)
                    Add(geometry);
            }
        }

        /// <summary>
        /// Adds the length defined by a coordinate.
        /// </summary>
        /// <param name="pt">A coordinate.</param>
        /// <remarks></remarks>
        public void Add(ICoordinate pt)
        {
            ptCount += 1;
            centSum.X += pt.X;
            centSum.Y += pt.Y;
        }

        /// <summary>
        /// Gets the centroid.
        /// </summary>
        /// <remarks></remarks>
        public ICoordinate Centroid
        {
            get
            {
                ICoordinate cent = new Coordinate();
                cent.X = centSum.X / ptCount;
                cent.Y = centSum.Y / ptCount;
                return cent;
            }
        }
    }   
}
