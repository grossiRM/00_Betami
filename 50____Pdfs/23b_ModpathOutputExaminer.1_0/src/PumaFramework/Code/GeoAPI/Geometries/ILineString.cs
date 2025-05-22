namespace GeoAPI.Geometries
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public interface ILineString : ICurve
    {
        /// <summary>
        /// Gets the point N.
        /// </summary>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IPoint GetPointN(int n);

        /// <summary>
        /// Gets the coordinate N.
        /// </summary>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        ICoordinate GetCoordinateN(int n);

        /// <summary>
        /// Reverses this instance.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        ILineString Reverse();
    }
}
