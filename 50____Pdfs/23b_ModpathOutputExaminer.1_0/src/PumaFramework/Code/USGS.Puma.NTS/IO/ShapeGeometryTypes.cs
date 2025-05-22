using System;
using System.Collections;
using System.Text;

namespace USGS.Puma.NTS.IO
{
    /// <summary>
    /// Feature type enumeration
    /// </summary>
    public enum ShapeGeometryTypes
    {
        // These values correspond to the ESRI shape types

        /// <summary>
        /// Null Shape
        /// </summary>
        NullShape = 0,

        /// <summary>
        /// Point
        /// </summary>
        Point = 1,

        /// <summary>
        /// LineString (ESRI: Polyline)
        /// </summary>
        LineString = 3,

        /// <summary>
        /// Polygon
        /// </summary>
        Polygon = 5,

        /// <summary>
        /// MultiPoint
        /// </summary>
        MultiPoint = 8,

        /// <summary>
        /// PointZ
        /// </summary>
        PointZ = 11,

        /// <summary>
        /// LineStringZ (ESRI: PolyLineZ)
        /// </summary>
        LineStringZ = 13,

        /// <summary>
        /// PolygonZ
        /// </summary>
        PolygonZ = 15,

        /// <summary>
        /// MultiPointZ
        /// </summary>
        MultiPointZ = 18,

        /// <summary>
        /// PointM
        /// </summary>
        PointM = 21,

        /// <summary>
        /// PolylineM
        /// </summary>
        LineStringM = 23,

        /// <summary>
        /// PolygonM
        /// </summary>
        PolygonM = 25,

        /// <summary>
        /// MultiPointM
        /// </summary>
        MultiPointM = 28,

        /// <summary>
        /// MultiPatch
        /// </summary>
        MultiPatch = 31,

        // Note*** The id values for PointZM, LimeStringZM, PolygonZM and MultiPointZM
        // were changed to avoid conflicts with the previous defined values that
        // match the ESRI shape type specs. 
        /// <summary>
        /// 
        /// </summary>
        PointZM = 35,

        /// <summary>
        /// 
        /// </summary>
        LineStringZM = 36,

        /// <summary>
        /// 
        /// </summary>
        PolygonZM = 37,

        /// <summary>
        /// 
        /// </summary>
        MultiPointZM

    }        
}
