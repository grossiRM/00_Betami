namespace GeoAPI.Geometries
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public interface IPoint : IGeometry
    {
        /// <summary>
        /// Gets or sets the X.
        /// </summary>
        /// <value>The X.</value>
        /// <remarks></remarks>
        double X { get; set; }

        /// <summary>
        /// Gets or sets the Y.
        /// </summary>
        /// <value>The Y.</value>
        /// <remarks></remarks>
        double Y { get; set; }

        /// <summary>
        /// Gets or sets the Z.
        /// </summary>
        /// <value>The Z.</value>
        /// <remarks></remarks>
        double Z { get; set; }

        /// <summary>
        /// Gets the coordinate sequence.
        /// </summary>
        /// <remarks></remarks>
        ICoordinateSequence CoordinateSequence { get; }
    }
}
