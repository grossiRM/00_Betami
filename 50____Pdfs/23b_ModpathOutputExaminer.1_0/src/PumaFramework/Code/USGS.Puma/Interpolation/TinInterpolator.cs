using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using GeoAPI.Geometries;
using USGS.Puma.NTS.Geometries;

namespace USGS.Puma.Interpolation
{
    /// <summary>
    /// Interpolates elevation and measure values associated with a point using
    /// a triangulated network of points (TIN).
    /// </summary>
    public class TinInterpolator
    {
        /// <summary>
        /// Initialize a TinInterpolator using the TriangulatedNetwork tin.
        /// </summary>
        /// <param name="tin"></param>
        public TinInterpolator(TriangulatedNetwork tin)
        {
            _Tin=tin;
        }

        private TriangulatedNetwork _Tin;
        /// <summary>
        /// Gets the trianulated network (TIN)
        /// </summary>
        public TriangulatedNetwork TriangulatedNetwork
        {
            get { return _Tin; }
        }

        private double _NoDataValue = 1.0E+30;
        /// <summary>
        /// Specified a value to assign to an elevation or measure when interpolation cannot be performed.
        /// </summary>
        public double NoDataValue
        {
            get { return _NoDataValue; }
            set { _NoDataValue = value; }
        }

        /// <summary>
        /// Interpolate elevation and(or) measure values associated with the point
        /// </summary>
        /// <param name="point"></param>
        /// <param name="interpolationOption"></param>
        /// <returns></returns>
        public void Interpolate(ICoordinateM point, CoordinateInterpolationOptions interpolationOption)
        {
            IPolygon triangle = LocateTriangle(point);
            switch (interpolationOption)
            {
                case CoordinateInterpolationOptions.InterpolateElevation:
                    InterpolateZ(point, triangle);
                    break;
                case CoordinateInterpolationOptions.InterpolateMeasure:
                    InterpolateM(point, triangle);
                    break;
                case CoordinateInterpolationOptions.InterpolateElevationAndMeasure:
                    InterpolateZ(point, triangle);
                    InterpolateM(point, triangle);
                    break;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="point"></param>
        protected virtual void InterpolateZ(ICoordinateM point, IPolygon triangle)
        {
            if (triangle == null)
            {
                point.Z = NoDataValue;
            }
            else
            {
                // Add code here
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="point"></param>
        protected virtual void InterpolateM(ICoordinateM point, IPolygon triangle)
        {
            if (triangle == null)
            {
                point.M = NoDataValue;
            }
            else
            {
                // Add code here
            }
        }
        /// <summary>
        /// Finds the triangle that contains the point
        /// </summary>
        /// <param name="point"></param>
        /// <returns></returns>
        protected virtual IPolygon LocateTriangle(ICoordinateM point)
        {
            throw new NotImplementedException();
        }

    }
}
