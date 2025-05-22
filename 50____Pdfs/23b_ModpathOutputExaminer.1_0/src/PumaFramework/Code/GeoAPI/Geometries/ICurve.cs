namespace GeoAPI.Geometries
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public interface ICurve : IGeometry
    {
        /// <summary>
        /// Gets the coordinate sequence.
        /// </summary>
        /// <remarks></remarks>
        ICoordinateSequence CoordinateSequence { get; }

        /// <summary>
        /// Gets the start point.
        /// </summary>
        /// <remarks></remarks>
        IPoint StartPoint { get; }

        /// <summary>
        /// Gets the end point.
        /// </summary>
        /// <remarks></remarks>
        IPoint EndPoint { get; }

        /// <summary>
        /// Gets a value indicating whether this instance is closed.
        /// </summary>
        /// <remarks></remarks>
        bool IsClosed { get; }

        /// <summary>
        /// Gets a value indicating whether this instance is ring.
        /// </summary>
        /// <remarks></remarks>
        bool IsRing { get; }        
    }
}
