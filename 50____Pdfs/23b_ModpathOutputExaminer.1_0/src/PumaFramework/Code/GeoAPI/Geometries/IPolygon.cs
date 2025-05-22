namespace GeoAPI.Geometries
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public interface IPolygon : ISurface
    {
        /// <summary>
        /// Gets the exterior ring.
        /// </summary>
        /// <remarks></remarks>
        ILineString ExteriorRing { get; }

        /// <summary>
        /// Gets the shell.
        /// </summary>
        /// <remarks></remarks>
        ILinearRing Shell { get; }

        /// <summary>
        /// Gets the num interior rings.
        /// </summary>
        /// <remarks></remarks>
        int NumInteriorRings { get; }

        /// <summary>
        /// Gets the interior rings.
        /// </summary>
        /// <remarks></remarks>
        ILineString[] InteriorRings { get; }

        /// <summary>
        /// Gets the interior ring N.
        /// </summary>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        ILineString GetInteriorRingN(int n);

        /// <summary>
        /// Gets the holes.
        /// </summary>
        /// <remarks></remarks>
        ILinearRing[] Holes { get; }  
    }
}
