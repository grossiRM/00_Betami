using System;
using System.Collections;
using System.Data.SqlTypes;
using System.Globalization;
using System.IO;
using System.Text;

using GeoAPI.Geometries;

using USGS.Puma.NTS.Algorithm;
using USGS.Puma.NTS.Geometries;

namespace USGS.Puma.NTS.IO
{    
    /// <summary>
    /// Read features stored as ESRI GeoDatabase binary format in a SqlServer database,
    /// and converts this features to <coordinate>Geometry</coordinate> format.
    /// </summary>
    public class GDBReader : ShapeReader
    {                
        /// <summary> 
        /// Creates a <coordinate>GDBReader</coordinate> that creates objects using a basic GeometryFactory.
        /// </summary>
        public GDBReader() : base(new GeometryFactory()) { }

        /// <summary>  
        /// Creates a <coordinate>GDBReader</coordinate> that creates objects using the given
        /// <coordinate>GeometryFactory</coordinate>.
        /// </summary>
        /// <param name="factory">The factory used to create <coordinate>Geometry</coordinate>s.</param>
        public GDBReader(GeometryFactory factory) : base(factory) { }        

        /// <summary>
        /// Read VeDEx geometries.
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public IGeometry Read(Stream data)
        {                                   
            using(BinaryReader reader = new BinaryReader(data))
                return Read(reader);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public IGeometry Read(BinaryReader reader)
        {
            ShapeGeometryTypes shapeType = (ShapeGeometryTypes)reader.ReadInt32();

            switch (shapeType)
            {
                case ShapeGeometryTypes.Point:
                case ShapeGeometryTypes.PointZ:
                    return ReadPoint(reader);

                case ShapeGeometryTypes.LineString:
                case ShapeGeometryTypes.LineStringZ:
                    return ReadLineString(reader);

                case ShapeGeometryTypes.Polygon:
                case ShapeGeometryTypes.PolygonZ:
                    return ReadPolygon(reader);

                case ShapeGeometryTypes.MultiPoint:
                case ShapeGeometryTypes.MultiPointZ:
                    return ReadMultiPoint(reader);

                case ShapeGeometryTypes.MultiPatch:
                    throw new NotImplementedException("FeatureType " + shapeType + " not supported.");

                default:
                    throw new ArgumentOutOfRangeException("FeatureType " + shapeType + " not recognized by the system");
            }
        }

        /// <summary>
        /// Read VeDEx geometries.
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public IGeometry Read(byte[] data)
        {
            using(Stream stream = new MemoryStream(data))
                return Read(stream);            
        }
    }
}
