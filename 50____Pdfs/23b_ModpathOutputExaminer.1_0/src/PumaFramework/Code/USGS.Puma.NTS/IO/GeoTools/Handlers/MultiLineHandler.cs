using System;
using System.Collections.Generic;

using GeoAPI.Geometries;

using USGS.Puma.NTS.Geometries;

namespace USGS.Puma.NTS.IO
{
	/// <summary>
	/// Converts a Shapefile multi-line to a OGIS LineString/MultiLineString.
	/// </summary>
	public class MultiLineHandler : ShapeHandler
	{
		/// <summary>
		/// Initializes a new instance of the MultiLineHandler class.
		/// </summary>
		public MultiLineHandler() : base() 
        {
            _ShapeType = ShapeGeometryTypes.LineStringZ;
        }
	
		/// <summary>
		/// Returns the ShapeType the handler handles.
		/// </summary>
        private ShapeGeometryTypes _ShapeType;
        public override ShapeGeometryTypes ShapeType
		{
			get
			{
                //return ShapeGeometryTypes.LineString;
                return _ShapeType;
			}
		}

		/// <summary>
		/// Reads a stream and converts the shapefile record to an equilivent geometry object.
		/// </summary>
		/// <param name="file">The stream to read.</param>
		/// <param name="geometryFactory">The geometry factory to use when making the object.</param>
		/// <returns>The Geometry object that represents the shape file record.</returns>
        public override IGeometry Read(BigEndianBinaryReader file, IGeometryFactory geometryFactory, int contentLength)
        {
            int shapeTypeNum = file.ReadInt32();
            ShapeGeometryTypes shapeType = (ShapeGeometryTypes)Enum.Parse(typeof(ShapeGeometryTypes), shapeTypeNum.ToString());
            if (!(shapeType == ShapeGeometryTypes.LineString || shapeType == ShapeGeometryTypes.LineStringM ||
                    shapeType == ShapeGeometryTypes.LineStringZ || shapeType == ShapeGeometryTypes.LineStringZM))
                throw new ShapefileException("Attempting to load a non-arc as arc.");

            //read and for now ignore bounds.
            double[] box = new double[4];
            for (int i = 0; i < 4; i++)
            {
                double d = file.ReadDouble();
                box[i] = d;
            }

            int numParts = file.ReadInt32();
            int numPoints = file.ReadInt32();
            int[] partOffsets = new int[numParts];
            for (int i = 0; i < numParts; i++)
                partOffsets[i] = file.ReadInt32();

            List<CoordinateList> coordList = new List<CoordinateList>();
            int start, finish, length;
            for (int part = 0; part < numParts; part++)
            {
                start = partOffsets[part];
                if (part == numParts - 1)
                    finish = numPoints;
                else finish = partOffsets[part + 1];
                length = finish - start;
                CoordinateList points = new CoordinateList();
                points.Capacity = length;
                ICoordinate external;
                for (int i = 0; i < length; i++)
                {
                    external = new Coordinate(file.ReadDouble(), file.ReadDouble());
                    geometryFactory.PrecisionModel.MakePrecise(external);
                    points.Add(external);
                }
                coordList.Add(points);
            }

            // Check to see if z values should be read and read them if
            // they are present.
            if (shapeType == ShapeGeometryTypes.LineStringZ)
            {
                double minZ = file.ReadDouble();
                double maxZ = file.ReadDouble();
                for (int part = 0; part < numParts; part++)
                {
                    CoordinateList coords = coordList[part];
                    for (int i = 0; i < coords.Count; i++)
                    {
                        coords[i].Z = file.ReadDouble();
                    }
                }

                // Check to see if the file contains M values. If so, read them.
                if (contentLength == this.GetLengthZM(numParts, numPoints))
                {
                    double minM = file.ReadDouble();
                    double maxM = file.ReadDouble();
                    for (int part = 0; part < numParts; part++)
                    {
                        CoordinateList coords = coordList[part];
                        for (int i = 0; i < coords.Count; i++)
                        {
                            (coords[i] as ICoordinateM).M = file.ReadDouble();
                        }
                    }
                }
                
            }

            // Create an MultiLineString using the coordinates prepared above and return the result
            ILineString[] lines = new ILineString[numParts];
            for (int part = 0; part < numParts; part++)
            {
                lines[part] = geometryFactory.CreateLineString(coordList[part].ToArray());
            }
            return geometryFactory.CreateMultiLineString(lines);
        }

		/// <summary>
		/// Writes to the given stream the equilivent shape file record given a Geometry object.
		/// </summary>
		/// <param name="geometry">The geometry object to write.</param>
		/// <param name="file">The stream to write to.</param>
		/// <param name="geometryFactory">The geometry factory to use.</param>
		public override void Write(IGeometry geometry, System.IO.BinaryWriter file, IGeometryFactory geometryFactory)
		{
			IMultiLineString multi = (IMultiLineString) geometry;
            file.Write(int.Parse(Enum.Format(typeof(ShapeGeometryTypes), this.ShapeType, "d")));
        
			IEnvelope box = multi.EnvelopeInternal;
			file.Write(box.MinX);
			file.Write(box.MinY);
			file.Write(box.MaxX);
			file.Write(box.MaxY);
        
			int numParts = multi.NumGeometries;
			int numPoints = multi.NumPoints;
        
			file.Write(numParts);		
			file.Write(numPoints);      
        
			// write the offsets
			int offset=0;
			for (int i = 0; i < numParts; i++)
			{
				IGeometry g = multi.GetGeometryN(i);
				file.Write( offset );
				offset = offset + g.NumPoints;
			}
        
			ICoordinate	external;
			for (int part = 0; part < numParts; part++)
			{
                CoordinateList points = new CoordinateList(multi.GetGeometryN(part).Coordinates);
				for (int i = 0; i < points.Count; i++)
				{
					external = points[i];
					file.Write(external.X);
					file.Write(external.Y);
				}
			}

            // Write data related to the Z values
            if (ShapeType == ShapeGeometryTypes.LineStringZ)
            {
                // Write Z envelope (min and max Z)
                IVerticalEnvelope ve = (multi as GeometryCollection).VerticalEnvelopeInternal;
                file.Write(ve.MinZ);
                file.Write(ve.MaxZ);

                double minM = 0.0;
                double maxM = 0.0;
                ICoordinateM c;

                // Write Z values for all the coordinates and use the point loop 
                // to compute the min and max M values.
                for (int part = 0; part < numParts; part++)
                {
                    CoordinateList points = new CoordinateList(multi.GetGeometryN(part).Coordinates);
                    for (int i = 0; i < points.Count; i++)
                    {
                        c = points[i] as ICoordinateM;
                        file.Write(c.Z);
                        if ((part == 0) && (i == 0))
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
                }

                // Write the min and max M values
                file.Write(minM);
                file.Write(maxM);
                // Write the M values for all the coordinates.
                for (int part = 0; part < numParts; part++)
                {
                    CoordinateList points = new CoordinateList(multi.GetGeometryN(part).Coordinates);
                    for (int i = 0; i < points.Count; i++)
                    {
                        c = points[i].CoordinateValue as ICoordinateM;
                        file.Write(c.M);
                    }
                }


                
            }

		}


		/// <summary>
		/// Gets the length in bytes the Geometry will need when written as a shape file record.
		/// </summary>
		/// <param name="geometry">The Geometry object to use.</param>
		/// <returns>The length in bytes the Geometry will use when represented as a shape file record.</returns>
		public override int GetLength(IGeometry geometry)
		{
            // The length value is expressed in terms of 2-byte WORDS
            int numParts = GetNumParts(geometry);
            int numPoints = geometry.NumPoints;
            return GetLengthZM(numParts, numPoints);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="numParts"></param>
        /// <param name="numPoints"></param>
        /// <returns></returns>
        public int GetLengthZM(int numParts, int numPoints)
        {
            // The length value is expressed in terms of 2-byte WORDS
            int nozLength = (22 + (2 * numParts) + numPoints * 8);
            int zLength = 8 + (4 * numPoints);
            int mLength = 8 + (4 * numPoints);
            int bodyLength = nozLength + zLength + mLength;
            return bodyLength;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="geometry"></param>
        /// <returns></returns>
		private int GetNumParts(IGeometry geometry)
		{
			int numParts=1;
			if (geometry is IMultiLineString)
				numParts = ((IMultiLineString) geometry).Geometries.Length;
			return numParts;
		}
	}
}