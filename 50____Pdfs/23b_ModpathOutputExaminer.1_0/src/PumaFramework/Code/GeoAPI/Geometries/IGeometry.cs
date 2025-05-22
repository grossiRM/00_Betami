using System;
using GeoAPI.Operations.Buffer;

namespace GeoAPI.Geometries
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public interface IGeometry : ICloneable, IComparable, IComparable<IGeometry>, IEquatable<IGeometry>
    {
        /// <summary>
        /// Gets the factory.
        /// </summary>
        /// <remarks></remarks>
        IGeometryFactory Factory { get; }
        /// <summary>
        /// Gets the precision model.
        /// </summary>
        /// <remarks></remarks>
        IPrecisionModel PrecisionModel { get; }

        /// <summary>
        /// Gets or sets the SRID.
        /// </summary>
        /// <value>The SRID.</value>
        /// <remarks></remarks>
        int SRID { get; set; }

        /// <summary>
        /// Gets the type of the geometry.
        /// </summary>
        /// <remarks></remarks>
        string GeometryType { get; }

        /// <summary>
        /// A ISurface method moved in IGeometry
        /// </summary>
        /// <remarks></remarks>
        double Area { get; }

        /// <summary>
        /// A ICurve method moved in IGeometry
        /// </summary>
        /// <remarks></remarks>
        double Length { get; }

        /// <summary>
        /// A IGeometryCollection method moved in IGeometry
        /// </summary>
        /// <remarks></remarks>
        int NumGeometries { get; }

        /// <summary>
        /// A ILineString method moved to IGeometry
        /// </summary>
        /// <remarks></remarks>
        int NumPoints { get; }

        /// <summary>
        /// Gets or sets the boundary.
        /// </summary>
        /// <value>The boundary.</value>
        /// <remarks></remarks>
        IGeometry Boundary { get; set; }

        /// <summary>
        /// Gets or sets the boundary dimension.
        /// </summary>
        /// <value>The boundary dimension.</value>
        /// <remarks></remarks>
        Dimensions BoundaryDimension { get; set; }

        /// <summary>
        /// A ISurface method moved in IGeometry
        /// </summary>
        /// <remarks></remarks>
        IPoint Centroid { get; }

        /// <summary>
        /// Gets the coordinate.
        /// </summary>
        /// <remarks></remarks>
        ICoordinate Coordinate { get; }

        /// <summary>
        /// Gets the coordinates.
        /// </summary>
        /// <remarks></remarks>
        ICoordinate[] Coordinates { get; }

        /// <summary>
        /// Gets or sets the dimension.
        /// </summary>
        /// <value>The dimension.</value>
        /// <remarks></remarks>
        Dimensions Dimension { get; set; }

        /// <summary>
        /// Gets the envelope.
        /// </summary>
        /// <remarks></remarks>
        IGeometry Envelope { get; }

        /// <summary>
        /// Gets the envelope internal.
        /// </summary>
        /// <remarks></remarks>
        IEnvelope EnvelopeInternal { get; }

        /// <summary>
        /// Gets the interior point.
        /// </summary>
        /// <remarks></remarks>
        IPoint InteriorPoint { get; }

        /// <summary>
        /// A ISurface method moved in IGeometry
        /// </summary>
        /// <remarks></remarks>
        IPoint PointOnSurface { get; }

        /// <summary>
        /// A IGeometryCollection method moved in IGeometry
        /// </summary>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IGeometry GetGeometryN(int n);

        /// <summary>
        /// Normalizes this instance.
        /// </summary>
        /// <remarks></remarks>
        void Normalize();

        /// <summary>
        /// Ases the binary.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        byte[] AsBinary();

        /// <summary>
        /// Ases the text.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        string AsText();

        /// <summary>
        /// Gets or sets the user data.
        /// </summary>
        /// <value>The user data.</value>
        /// <remarks></remarks>
        object UserData { get; set; }

        /// <summary>
        /// Convexes the hull.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        IGeometry ConvexHull();

        /// <summary>
        /// Relates the specified g.
        /// </summary>
        /// <param name="g">The g.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IntersectionMatrix Relate(IGeometry g);

        /// <summary>
        /// Differences the specified other.
        /// </summary>
        /// <param name="other">The other.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IGeometry Difference(IGeometry other);

        /// <summary>
        /// Symmetrics the difference.
        /// </summary>
        /// <param name="other">The other.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IGeometry SymmetricDifference(IGeometry other);

        /// <summary>
        /// Buffers the specified distance.
        /// </summary>
        /// <param name="distance">The distance.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IGeometry Buffer(double distance);

        /// <summary>
        /// Buffers the specified distance.
        /// </summary>
        /// <param name="distance">The distance.</param>
        /// <param name="quadrantSegments">The quadrant segments.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IGeometry Buffer(double distance, int quadrantSegments);

        /// <summary>
        /// Buffers the specified distance.
        /// </summary>
        /// <param name="distance">The distance.</param>
        /// <param name="endCapStyle">The end cap style.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IGeometry Buffer(double distance, BufferStyle endCapStyle);

        /// <summary>
        /// Buffers the specified distance.
        /// </summary>
        /// <param name="distance">The distance.</param>
        /// <param name="quadrantSegments">The quadrant segments.</param>
        /// <param name="endCapStyle">The end cap style.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IGeometry Buffer(double distance, int quadrantSegments, BufferStyle endCapStyle);

        /// <summary>
        /// Intersections the specified other.
        /// </summary>
        /// <param name="other">The other.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IGeometry Intersection(IGeometry other);

        /// <summary>
        /// Unions the specified other.
        /// </summary>
        /// <param name="other">The other.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IGeometry Union(IGeometry other);

        /// <summary>
        /// Equalses the exact.
        /// </summary>
        /// <param name="other">The other.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool EqualsExact(IGeometry other);

        /// <summary>
        /// Equalses the exact.
        /// </summary>
        /// <param name="other">The other.</param>
        /// <param name="tolerance">The tolerance.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool EqualsExact(IGeometry other, double tolerance);

        /// <summary>
        /// Gets a value indicating whether this instance is empty.
        /// </summary>
        /// <remarks></remarks>
        bool IsEmpty { get; }

        /// <summary>
        /// Gets a value indicating whether this instance is rectangle.
        /// </summary>
        /// <remarks></remarks>
        bool IsRectangle { get; }

        /// <summary>
        /// Gets a value indicating whether this instance is simple.
        /// </summary>
        /// <remarks></remarks>
        bool IsSimple { get; }

        /// <summary>
        /// Gets a value indicating whether this instance is valid.
        /// </summary>
        /// <remarks></remarks>
        bool IsValid { get; }

        /// <summary>
        /// Withins the specified g.
        /// </summary>
        /// <param name="g">The g.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Within(IGeometry g);

        /// <summary>
        /// Determines whether [contains] [the specified g].
        /// </summary>
        /// <param name="g">The g.</param>
        /// <returns><c>true</c> if [contains] [the specified g]; otherwise, <c>false</c>.</returns>
        /// <remarks></remarks>
        bool Contains(IGeometry g);

        /// <summary>
        /// Determines whether [is within distance] [the specified geom].
        /// </summary>
        /// <param name="geom">The geom.</param>
        /// <param name="distance">The distance.</param>
        /// <returns><c>true</c> if [is within distance] [the specified geom]; otherwise, <c>false</c>.</returns>
        /// <remarks></remarks>
        bool IsWithinDistance(IGeometry geom, double distance);

        /// <summary>
        /// Covereds the by.
        /// </summary>
        /// <param name="g">The g.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool CoveredBy(IGeometry g);

        /// <summary>
        /// Coverses the specified g.
        /// </summary>
        /// <param name="g">The g.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Covers(IGeometry g);

        /// <summary>
        /// Crosseses the specified g.
        /// </summary>
        /// <param name="g">The g.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Crosses(IGeometry g);

        /// <summary>
        /// Intersectses the specified g.
        /// </summary>
        /// <param name="g">The g.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Intersects(IGeometry g);

        /// <summary>
        /// Overlapses the specified g.
        /// </summary>
        /// <param name="g">The g.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Overlaps(IGeometry g);

        /// <summary>
        /// Relates the specified g.
        /// </summary>
        /// <param name="g">The g.</param>
        /// <param name="intersectionPattern">The intersection pattern.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Relate(IGeometry g, string intersectionPattern);

        /// <summary>
        /// Toucheses the specified g.
        /// </summary>
        /// <param name="g">The g.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Touches(IGeometry g);

        /// <summary>
        /// Disjoints the specified g.
        /// </summary>
        /// <param name="g">The g.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Disjoint(IGeometry g);

        /// <summary>
        /// Distances the specified g.
        /// </summary>
        /// <param name="g">The g.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        double Distance(IGeometry g);

        /// <summary>
        /// Applies the specified filter.
        /// </summary>
        /// <param name="filter">The filter.</param>
        /// <remarks></remarks>
        void Apply(ICoordinateFilter filter);

        /// <summary>
        /// Applies the specified filter.
        /// </summary>
        /// <param name="filter">The filter.</param>
        /// <remarks></remarks>
        void Apply(IGeometryFilter filter);

        /// <summary>
        /// Applies the specified filter.
        /// </summary>
        /// <param name="filter">The filter.</param>
        /// <remarks></remarks>
        void Apply(IGeometryComponentFilter filter);

        /// <summary>
        /// Geometries the changed.
        /// </summary>
        /// <remarks></remarks>
        void GeometryChanged();

        /// <summary>
        /// Geometries the changed action.
        /// </summary>
        /// <remarks></remarks>
        void GeometryChangedAction();
    }
}
