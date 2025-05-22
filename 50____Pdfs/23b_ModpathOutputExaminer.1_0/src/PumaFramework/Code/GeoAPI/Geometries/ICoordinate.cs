using System;

namespace GeoAPI.Geometries
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public interface ICoordinate : ICloneable, IComparable, IComparable<ICoordinate>, IEquatable<ICoordinate>
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
        /// Gets or sets the coordinate value.
        /// </summary>
        /// <value>The coordinate value.</value>
        /// <remarks></remarks>
        ICoordinate CoordinateValue { get; set; }

        /// <summary>
        /// Distances the specified p.
        /// </summary>
        /// <param name="p">The p.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        double Distance(ICoordinate p);

        /// <summary>
        /// Equals2s the D.
        /// </summary>
        /// <param name="other">The other.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Equals2D(ICoordinate other);

        /// <summary>
        /// Equals3s the D.
        /// </summary>
        /// <param name="other">The other.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Equals3D(ICoordinate other);        
    }
}
