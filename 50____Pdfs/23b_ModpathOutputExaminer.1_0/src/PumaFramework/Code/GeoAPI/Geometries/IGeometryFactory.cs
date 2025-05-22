using System.Collections;

namespace GeoAPI.Geometries
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public interface IGeometryFactory
    {
        /// <summary>
        /// Gets the coordinate sequence factory.
        /// </summary>
        /// <remarks></remarks>
        ICoordinateSequenceFactory CoordinateSequenceFactory { get; }

        /// <summary>
        /// Gets the SRID.
        /// </summary>
        /// <remarks></remarks>
        int SRID { get; }
        /// <summary>
        /// Gets the precision model.
        /// </summary>
        /// <remarks></remarks>
        IPrecisionModel PrecisionModel { get; }

        /// <summary>
        /// Builds the geometry.
        /// </summary>
        /// <param name="geomList">The geom list.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IGeometry BuildGeometry(ICollection geomList);
        /// <summary>
        /// Creates the geometry.
        /// </summary>
        /// <param name="g">The g.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IGeometry CreateGeometry(IGeometry g);

        /// <summary>
        /// Creates the point.
        /// </summary>
        /// <param name="coordinate">The coordinate.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IPoint CreatePoint(ICoordinate coordinate);
        /// <summary>
        /// Creates the point.
        /// </summary>
        /// <param name="coordinates">The coordinates.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IPoint CreatePoint(ICoordinateSequence coordinates);

        /// <summary>
        /// Creates the line string.
        /// </summary>
        /// <param name="coordinates">The coordinates.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        ILineString CreateLineString(ICoordinate[] coordinates);
        /// <summary>
        /// Creates the line string.
        /// </summary>
        /// <param name="coordinates">The coordinates.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        ILineString CreateLineString(ICoordinateSequence coordinates);

        /// <summary>
        /// Creates the linear ring.
        /// </summary>
        /// <param name="coordinates">The coordinates.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        ILinearRing CreateLinearRing(ICoordinate[] coordinates);
        /// <summary>
        /// Creates the linear ring.
        /// </summary>
        /// <param name="coordinates">The coordinates.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        ILinearRing CreateLinearRing(ICoordinateSequence coordinates);

        /// <summary>
        /// Creates the polygon.
        /// </summary>
        /// <param name="shell">The shell.</param>
        /// <param name="holes">The holes.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IPolygon CreatePolygon(ILinearRing shell, ILinearRing[] holes);

        /// <summary>
        /// Creates the multi point.
        /// </summary>
        /// <param name="coordinates">The coordinates.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IMultiPoint CreateMultiPoint(ICoordinate[] coordinates);
        /// <summary>
        /// Creates the multi point.
        /// </summary>
        /// <param name="point">The point.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IMultiPoint CreateMultiPoint(IPoint[] point);
        /// <summary>
        /// Creates the multi point.
        /// </summary>
        /// <param name="coordinates">The coordinates.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IMultiPoint CreateMultiPoint(ICoordinateSequence coordinates);

        /// <summary>
        /// Creates the multi line string.
        /// </summary>
        /// <param name="lineStrings">The line strings.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IMultiLineString CreateMultiLineString(ILineString[] lineStrings);

        /// <summary>
        /// Creates the multi polygon.
        /// </summary>
        /// <param name="polygons">The polygons.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IMultiPolygon CreateMultiPolygon(IPolygon[] polygons);

        /// <summary>
        /// Creates the geometry collection.
        /// </summary>
        /// <param name="geometries">The geometries.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IGeometryCollection CreateGeometryCollection(IGeometry[] geometries);

        /// <summary>
        /// Toes the geometry.
        /// </summary>
        /// <param name="envelopeInternal">The envelope internal.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IGeometry ToGeometry(IEnvelope envelopeInternal);
    }
}
