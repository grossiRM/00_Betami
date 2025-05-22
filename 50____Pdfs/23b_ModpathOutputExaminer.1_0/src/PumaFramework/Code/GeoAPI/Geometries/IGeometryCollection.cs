using System.Collections;

namespace GeoAPI.Geometries
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public interface IGeometryCollection : IGeometry, IEnumerable
    {
        /// <summary>
        /// Gets the count.
        /// </summary>
        /// <remarks></remarks>
        int Count { get; }

        /// <summary>
        /// Gets the geometries.
        /// </summary>
        /// <remarks></remarks>
        IGeometry[] Geometries { get; }

        /// <summary>
        /// Gets the <see cref="GeoAPI.Geometries.IGeometry"/> with the specified i.
        /// </summary>
        /// <remarks></remarks>
        IGeometry this[int i] { get; }

        /// <summary>
        /// Gets a value indicating whether this instance is homogeneous.
        /// </summary>
        /// <remarks></remarks>
        bool IsHomogeneous { get; }                
    }
}
