using System;
using System.Collections;
using System.Diagnostics;

using GeoAPI.Geometries;

using USGS.Puma.NTS.Algorithm;
using USGS.Puma.NTS.Geometries;

namespace USGS.Puma.NTS.IO
{
	/// <summary>
	/// Converts a Shapefile point to a OGIS Polygon.
	/// </summary>
	public class MultiPointHandler : ShapeHandler
	{		
		/// <summary>
        /// Initializes a new instance of the MultiPointHandler class.
		/// </summary>
		public MultiPointHandler() 
        {
            _ShapeType = ShapeGeometryTypes.MultiPointZ;
        }

		/// <summary>
		/// The ShapeType this handler handles.
		/// </summary>
        private ShapeGeometryTypes _ShapeType;
        public override ShapeGeometryTypes ShapeType
		{
			get
			{
                return _ShapeType;
			}
		}		

		/// <summary>
		/// Reads a stream and converts the shapefile record to an equilivant geometry object.
		/// </summary>
		/// <param name="file">The stream to read.</param>
		/// <param name="geometryFactory">The geometry factory to use when making the object.</param>
		/// <returns>The Geometry object that represents the shape file record.</returns>
        public override IGeometry Read(BigEndianBinaryReader file, IGeometryFactory geometryFactory, int contentLength)
        {
            int shapeTypeNum = file.ReadInt32();
            ShapeGeometryTypes shapeType = (ShapeGeometryTypes)Enum.Parse(typeof(ShapeGeometryTypes), shapeTypeNum.ToString());
            if (!(shapeType == ShapeGeometryTypes.MultiPoint || shapeType == ShapeGeometryTypes.MultiPointM ||
                     shapeType == ShapeGeometryTypes.MultiPointZ || shapeType == ShapeGeometryTypes.MultiPointZM))
                throw new ShapefileException("Attempting to load a non-multipoint as multipoint.");

            // Read and for now ignore bounds.
            double[] box = new double[4];
            for (int i = 0; i < 4; i++)
                box[i] = file.ReadDouble();

            // Read points
            int numPoints = file.ReadInt32();
            IPoint[] points = new IPoint[numPoints];
            ICoordinate[] coords = new ICoordinate[numPoints];
            for (int i = 0; i < numPoints; i++)
            {
                coords[i] = new Coordinate(file.ReadDouble(), file.ReadDouble());
            }

            // Check to see if z values are present and read them if they are.
            if (shapeType == ShapeGeometryTypes.MultiPointZ)
            {
                double minZ = file.ReadDouble();
                double maxZ = file.ReadDouble();
                for (int i = 0; i < numPoints; i++)
                {
                    coords[i].Z = file.ReadDouble();
                }

                // Check for M values and read them if present
                if (contentLength == this.GetLengthZM(numPoints))
                {
                    double minM = file.ReadDouble();
                    double maxM = file.ReadDouble();
                    for (int i = 0; i < numPoints; i++)
                    {
                        (coords[i] as ICoordinateM).M = file.ReadDouble();
                    }
                }
            }

            // Create MultiPoint and return
            for (int i = 0; i < numPoints; i++)
            {
                points[i] = geometryFactory.CreatePoint(coords[i]);
            }
            return geometryFactory.CreateMultiPoint(points);
        }

		/// <summary>
		/// Writes a Geometry to the given binary wirter.
		/// </summary>
		/// <param name="geometry">The geometry to write.</param>
		/// <param name="file">The file stream to write to.</param>
		/// <param name="geometryFactory">The geometry factory to use.</param>
		public override void Write(IGeometry geometry, System.IO.BinaryWriter file, IGeometryFactory geometryFactory)
		{
            if(!(geometry is IMultiPoint))
                throw new ArgumentException("Geometry Type error: MultiPoint expected, but the type retrieved is " + geometry.GetType().Name);

            // Slow and maybe not useful...
			// if (!geometry.IsValid)
			// 	Trace.WriteLine("Invalid multipoint being written.");

            IMultiPoint mpoint = geometry as IMultiPoint;
            
            file.Write(int.Parse(Enum.Format(typeof(ShapeGeometryTypes), this.ShapeType, "d")));

            IEnvelope box = geometry.EnvelopeInternal;
			IEnvelope bounds = ShapeHandler.GetEnvelopeExternal(geometryFactory.PrecisionModel, box);
			file.Write(bounds.MinX);
			file.Write(bounds.MinY);
			file.Write(bounds.MaxX);
			file.Write(bounds.MaxY);

            int numPoints = mpoint.NumPoints;
			file.Write(numPoints);						

			// write the points 
			for (int i = 0; i < numPoints; i++)
			{
                IPoint point = (IPoint) mpoint.Geometries[i];
                file.Write(point.X);
                file.Write(point.Y);	
			}
            
            // Write data for supporting Z values
            if (this.ShapeType == ShapeGeometryTypes.MultiPointZ)
            {
                IVerticalEnvelope ve = (mpoint as GeometryCollection).VerticalEnvelopeInternal;
                double minM = 0.0;
                double maxM = 0.0;
                // Write Z data
                file.Write(ve.MinZ);
                file.Write(ve.MaxZ);
                ICoordinateM c;
                for (int i = 0; i < numPoints; i++)
                {
                    IPoint point = (IPoint)mpoint.Geometries[i];
                    file.Write(point.Z);
                    c = point.Coordinate as ICoordinateM;
                    if (i == 0)
                    {
                        minM = c.M;
                        maxM = c.M;
                    }
                    else
                    {
                        if (c.M < minM)
                        { minM = c.M; }
                        if (c.M > maxM)
                        { maxM = c.M; }
                    }
                }
                // Write M data
                file.Write(minM);
                file.Write(maxM);
                for (int i = 0; i < numPoints; i++)
                {
                    c = mpoint.Coordinates[i] as ICoordinateM;
                    file.Write(c.M);
                }


            }

		}
		
		/// <summary>
		/// Gets the length of the shapefile record using the geometry passed in.
		/// </summary>
		/// <param name="geometry">The geometry to get the length for.</param>
		/// <returns>The length in bytes this geometry is going to use when written out as a shapefile record.</returns>
		public override int GetLength(IGeometry geometry)
		{
            return GetLengthZM(geometry.NumPoints);
		}

        public int GetLengthZM(int numPoints)
        {
            int nozLength = 22 + numPoints * 8;
            int zLength = 8 + (4 * numPoints);
            int mLength = 8 + (4 * numPoints);
            int bodyLength = nozLength + zLength + mLength;
            return bodyLength;
        }
	}
}