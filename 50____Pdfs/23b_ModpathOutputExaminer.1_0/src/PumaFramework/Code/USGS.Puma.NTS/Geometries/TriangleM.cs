using System;
using System.Collections.Generic;
using System.Text;
using GeoAPI.Geometries;

namespace USGS.Puma.NTS.Geometries
{
    public class TriangleM
    {
        private ICoordinateM p0, p1, p2;
        private ICoordinateM oldP0, oldP1, oldP2;
        private double az, am;
        private double bz, bm;
        private double dz, dm;

        /// <summary>
        /// 
        /// </summary>
        public ICoordinateM P0
        {
            get { return p0; }
        }

        /// <summary>
        /// 
        /// </summary>
        public ICoordinateM P1
        {
            get { return p1; }
        }

        /// <summary>
        /// 
        /// </summary>
        public ICoordinateM P2
        {
            get { return p2; }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="p0"></param>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        public TriangleM(ICoordinateM p0, ICoordinateM p1, ICoordinateM p2)
        {
            this.p0 = new Coordinate(p0);
            this.p1 = new Coordinate(p1);
            this.p2 = new Coordinate(p2);
            this.oldP0 = new Coordinate(p0);
            this.oldP1 = new Coordinate(p1);
            this.oldP2 = new Coordinate(p2);
            ComputeInterpolationParameters();
        }

        /// <summary>
        /// The inCentre of a triangle is the point which is equidistant
        /// from the sides of the triangle.  This is also the point at which the bisectors
        /// of the angles meet.
        /// </summary>
        /// <returns>
        /// The point which is the InCentre of the triangle.
        /// </returns>
        public ICoordinateM InCentre
        {
            get
            {
                // the lengths of the sides, labelled by their opposite vertex
                double len0 = P1.Distance(P2);
                double len1 = P0.Distance(P2);
                double len2 = P0.Distance(P1);
                double circum = len0 + len1 + len2;

                double inCentreX = (len0 * P0.X + len1 * P1.X + len2 * P2.X) / circum;
                double inCentreY = (len0 * P0.Y + len1 * P1.Y + len2 * P2.Y) / circum;
                return new Coordinate(inCentreX, inCentreY);
            }
        }

        /// <summary>
        /// Computes the Z and M values for the input point's X and Y values that
        /// correspond to a point on the plane defined by the triangle.
        /// </summary>
        /// <param name="point"></param>
        public void ComputePointOnPlane(ICoordinateM point)
        {
            if (!p0.Equals3DM(oldP0) || !p1.Equals3DM(oldP1) || !p2.Equals3DM(oldP2))
            {
                ComputeInterpolationParameters();
            }
            point.Z = -az * point.X - bz * point.Y - dz;
            point.M = -am * point.X - bm * point.Y - dm;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="point"></param>
        public void ComputePointOnPlaneXYZ(ICoordinateM point)
        {
            if (!p0.Equals3DM(oldP0) || !p1.Equals3DM(oldP1) || !p2.Equals3DM(oldP2))
            {
                ComputeInterpolationParameters();
            }
            point.Z = -az * point.X - bz * point.Y - dz;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="point"></param>
        public void ComputePointOnPlaneXYM(ICoordinateM point)
        {
            if (!p0.Equals3DM(oldP0) || !p1.Equals3DM(oldP1) || !p2.Equals3DM(oldP2))
            {
                ComputeInterpolationParameters();
            }
            point.M = -am * point.X - bm * point.Y - dm;
        }
        /// <summary>
        /// 
        /// </summary>
        private void ComputeInterpolationParameters()
        {
            double c;
            az = (p0.Y * (p1.Z - p2.Z)) + (p1.Y * (p2.Z - p0.Z)) + (p2.Y * (p0.Z - p1.Z));
            bz = (p0.Z * (p1.X - p2.X)) + (p1.Z * (p2.X - p0.X)) + (p2.Z * (p0.X - p1.X));
            c =  (p0.X * (p1.Y - p2.Y)) + (p1.X * (p2.Y - p0.Y)) + (p2.X * (p0.Y - p1.Y));
            dz = -az * p0.X - bz * p0.Y - c * p0.Z;
            az = az / c;
            bz = bz / c;
            dz = dz / c;

            am = (p0.Y * (p1.M - p2.M)) + (p1.Y * (p2.M - p0.M)) + (p2.Y * (p0.M - p1.M));
            bm = (p0.M * (p1.X - p2.X)) + (p1.M * (p2.X - p0.X)) + (p2.M * (p0.X - p1.X));
            dm = -am * p0.X - bm * p0.Y - c * p0.Z;
            am = am / c;
            bm = bm / c;
            dm = dm / c;

            oldP0.X = p0.X;
            oldP0.Y = p0.Y;
            oldP0.Z = p0.Z;
            oldP0.M = p0.M;

            oldP1.X = p1.X;
            oldP1.Y = p1.Y;
            oldP1.Z = p1.Z;
            oldP1.M = p1.M;

            oldP2.X = p2.X;
            oldP2.Y = p2.Y;
            oldP2.Z = p2.Z;
            oldP2.M = p2.M;

        }
    }
}
