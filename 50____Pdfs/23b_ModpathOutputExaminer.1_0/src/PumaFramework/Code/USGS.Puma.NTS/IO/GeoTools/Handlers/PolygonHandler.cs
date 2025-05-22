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
	public class PolygonHandler : ShapeHandler
	{		
		/// <summary>
		/// Initializes a new instance of the PolygonHandler class.
		/// </summary>
		public PolygonHandler() 
        {
            _ShapeType = ShapeGeometryTypes.PolygonZ;
        }

		/// <summary>
		/// The ShapeType this handler handles.
		/// </summary>
        private ShapeGeometryTypes _ShapeType = ShapeGeometryTypes.Polygon;
        public override ShapeGeometryTypes ShapeType
		{
			get
			{
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
            if ( ! ( shapeType == ShapeGeometryTypes.Polygon  || shapeType == ShapeGeometryTypes.PolygonM ||
                     shapeType == ShapeGeometryTypes.PolygonZ || shapeType == ShapeGeometryTypes.PolygonZM))	
				throw new ShapefileException("Attempting to load a non-polygon as polygon.");

			// Read and for now ignore bounds.
			double[] box = new double[4];
			for (int i = 0; i < 4; i++) 
				box[i] = file.ReadDouble();

			int[] partOffsets;        
			int numParts = file.ReadInt32();
			int numPoints = file.ReadInt32();
			partOffsets = new int[numParts];
			for (int i = 0; i < numParts; i++)
				partOffsets[i] = file.ReadInt32();

			ArrayList shells = new ArrayList();
			ArrayList holes = new ArrayList();

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
				for (int i = 0; i < length; i++)
				{
					ICoordinate external = new Coordinate(file.ReadDouble(), file.ReadDouble() );					
                    geometryFactory.PrecisionModel.MakePrecise( external);
                    ICoordinate internalCoord = external;

                    // Thanks to Abhay Menon!
                    if (!Double.IsNaN(internalCoord.Y) && !Double.IsNaN(internalCoord.X))
                         points.Add(internalCoord, false);
 				}

                if (points.Count > 2) // Thanks to Abhay Menon!
                {
                    try
                    {
                        if (points[0].Distance(points[points.Count - 1]) > .00001)
                            points.Add(new Coordinate(points[0]));
                        else if (points[0].Distance(points[points.Count - 1]) > 0.0)
                            points[points.Count - 1].CoordinateValue = points[0];

                        ILinearRing ring = geometryFactory.CreateLinearRing(points.ToArray());

                        // If shape have only a part, jump orientation check and add to shells
                        if (numParts == 1)
                            shells.Add(ring);
                        else
                        {
                            // Orientation check
                            if (CGAlgorithms.IsCCW(points.ToArray()))
                                holes.Add(ring);
                            else shells.Add(ring);
                        }
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine(ex.Message);
                    }
                }
			}

			// Now we have a list of all shells and all holes
			ArrayList holesForShells = new ArrayList(shells.Count);
			for (int i = 0; i < shells.Count; i++)
				holesForShells.Add(new ArrayList());
			// Find holes
			for (int i = 0; i < holes.Count; i++)
			{
				ILinearRing testRing = (ILinearRing) holes[i];
				ILinearRing minShell = null;
				IEnvelope minEnv = null;
                IEnvelope testEnv = testRing.EnvelopeInternal;
                ICoordinate testPt = testRing.GetCoordinateN(0);
				ILinearRing tryRing;
				for (int j = 0; j < shells.Count; j++)
				{
					tryRing = (ILinearRing) shells[j];
                    IEnvelope tryEnv = tryRing.EnvelopeInternal;
					if (minShell != null)
                        minEnv = minShell.EnvelopeInternal;
					bool isContained = false;
					CoordinateList coordList = new CoordinateList(tryRing.Coordinates);
					if (tryEnv.Contains(testEnv)
                        && (CGAlgorithms.IsPointInRing(testPt, coordList.ToArray()) 
                        || (PointInList(testPt, coordList)))) 				
						isContained = true;

                    // Check if this new containing ring is smaller than the current minimum ring
                    if (isContained)
                    {
                        if (minShell == null || minEnv.Contains(tryEnv))
                            minShell = tryRing;             

                        // Suggested by Brian Macomber and added 3/28/2006:
                        // holes were being found but never added to the holesForShells array
                        // so when converted to geometry by the factory, the inner rings were never created.
                        ArrayList holesForThisShell = (ArrayList) holesForShells[j];
                        holesForThisShell.Add(testRing);
                    }
				}
			}

            // Check to see if z values are present and read them if they are.
            if (shapeType == ShapeGeometryTypes.PolygonZ)
            {
                double minZ = file.ReadDouble();
                double maxZ = file.ReadDouble();
                for (int i = 0; i < shells.Count; i++)
                {
                    ILinearRing ring = shells[i] as ILinearRing;
                    ICoordinate[] coords = ring.Coordinates;
                    for (int j = 0; j < coords.Length; j++)
                    {
                        coords[j].Z = file.ReadDouble();
                    }
                }
                // Check for M values and read them if they are present
                if (contentLength == this.GetLengthZM(numParts, numPoints))
                {
                    double minM = file.ReadDouble();
                    double maxM = file.ReadDouble();
                    for (int i = 0; i < shells.Count; i++)
                    {
                        ILinearRing ring = shells[i] as ILinearRing;
                        ICoordinate[] coords = ring.Coordinates;
                        for (int j = 0; j < coords.Length; j++)
                        {
                            (coords[j] as ICoordinateM).M = file.ReadDouble();
                        }
                    }
                }
            }

			IPolygon[] polygons = new IPolygon[shells.Count];
			for (int i = 0; i < shells.Count; i++)
                polygons[i] = (geometryFactory.CreatePolygon((ILinearRing) shells[i], 
                    (ILinearRing[]) ((ArrayList)holesForShells[i]).ToArray(typeof(ILinearRing))));
        
			if (polygons.Length == 1)
				return polygons[0];
			// It's a multi part
			return geometryFactory.CreateMultiPolygon(polygons);
		}

		/// <summary>
		/// Writes a Geometry to the given binary wirter.
		/// </summary>
		/// <param name="geometry">The geometry to write.</param>
		/// <param name="file">The file stream to write to.</param>
		/// <param name="geometryFactory">The geometry factory to use.</param>
		public override void Write(IGeometry geometry, System.IO.BinaryWriter file, IGeometryFactory geometryFactory)
		{
            // This check seems to be not useful and slow the operations...
			// if (!geometry.IsValid)    
			// Trace.WriteLine("Invalid polygon being written.");

			IGeometryCollection multi = null;
			if (geometry is IGeometryCollection)
				multi = (IGeometryCollection) geometry;
			else 
			{
				GeometryFactory gf = new GeometryFactory(geometry.PrecisionModel);				
				multi = gf.CreateMultiPolygon(new IPolygon[] { (IPolygon) geometry, } );
			}

			file.Write(int.Parse(Enum.Format(typeof(ShapeGeometryTypes), this.ShapeType, "d")));

            IEnvelope box = multi.EnvelopeInternal;
			IEnvelope bounds = ShapeHandler.GetEnvelopeExternal(geometryFactory.PrecisionModel,  box);
			file.Write(bounds.MinX);
			file.Write(bounds.MinY);
			file.Write(bounds.MaxX);
			file.Write(bounds.MaxY);
        
			int numParts = GetNumParts(multi);
			int numPoints = multi.NumPoints;
			file.Write(numParts);
			file.Write(numPoints);
        			
			// write the offsets to the points
			int offset=0;
			for (int part = 0; part < multi.NumGeometries; part++)
			{
				// offset to the shell points
				IPolygon polygon = (IPolygon) multi.Geometries[part];
				file.Write(offset);
				offset = offset + polygon.ExteriorRing.NumPoints;

				// offstes to the holes
				foreach (ILinearRing ring in polygon.InteriorRings)
				{
					file.Write(offset);
					offset = offset + ring.NumPoints;
				}	
			}

			// write the points 
			for (int part = 0; part < multi.NumGeometries; part++)
			{
                IPolygon poly = (IPolygon) multi.Geometries[part];
				ICoordinate[] points = poly.ExteriorRing.Coordinates;
				WriteCoords(new CoordinateList(points), file, geometryFactory);
				foreach(ILinearRing ring in poly.InteriorRings)
				{
                    ICoordinate[] points2 = ring.Coordinates;					
					WriteCoords(new CoordinateList(points2), file, geometryFactory);
				}
			}

            // Write data for supporting Z values
            if (this.ShapeType == ShapeGeometryTypes.PolygonZ)
            {
                GeometryCollection g = multi as GeometryCollection;
                IVerticalEnvelope ve = g.VerticalEnvelopeInternal;
                double minM = 0.0;
                double maxM = 0.0;

                // Write Z data
                file.Write(ve.MinZ);
                file.Write(ve.MaxZ);
                ICoordinate[] coords = g.Coordinates;
                ICoordinateM c;
                for (int i = 0; i < coords.Length; i++)
                {
                    c = coords[i] as ICoordinateM;
                    file.Write(c.Z);
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
                for (int i = 0; i < coords.Length; i++)
                {
                    c = coords[i] as ICoordinateM;
                    file.Write(c.M);
                }

            }

		}

        /// <summary>
        /// 
        /// </summary>
        /// <param name="points"></param>
        /// <param name="file"></param>
        /// <param name="geometryFactory"></param>
		private void WriteCoords(CoordinateList points, System.IO.BinaryWriter file, IGeometryFactory geometryFactory)
		{
			ICoordinate external;
			foreach (ICoordinate point in points)
			{
				// external = geometryFactory.PrecisionModel.ToExternal(point);
                external = point;
				file.Write(external.X);
				file.Write(external.Y);
			}
		}

		/// <summary>
		/// Gets the length of the shapefile record using the geometry passed in.
		/// </summary>
		/// <param name="geometry">The geometry to get the length for.</param>
		/// <returns>The length in bytes this geometry is going to use when written out as a shapefile record.</returns>
		public override int GetLength(IGeometry geometry)
		{
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
            int nozLength = 22 + (2 * numParts) + numPoints * 8;
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
			int numParts = 0;
			if (geometry is IMultiPolygon)
            {
                IMultiPolygon mpoly = geometry as IMultiPolygon;
                foreach (IPolygon poly in mpoly.Geometries)
					numParts = numParts + poly.InteriorRings.Length + 1;
            }
			else if (geometry is IPolygon)
				numParts = ((IPolygon) geometry).InteriorRings.Length + 1;
			else throw new InvalidOperationException("Should not get here.");
			return numParts;
		}

		/// <summary>
		/// Test if a point is in a list of coordinates.
		/// </summary>
		/// <param name="testPoint">TestPoint the point to test for.</param>
		/// <param name="pointList">PointList the list of points to look through.</param>
		/// <returns>true if testPoint is a point in the pointList list.</returns>
		private bool PointInList(ICoordinate testPoint, CoordinateList pointList) 
		{
			foreach(ICoordinate p in pointList)
				if (p.Equals2D(testPoint))
					return true;
			return false;
		}
	}
}
