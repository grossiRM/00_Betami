using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace GeoAPI.Geometries
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public interface ICoordinateM : ICoordinate
    {
        /// <summary>
        /// Gets or sets the M.
        /// </summary>
        /// <value>The M.</value>
        /// <remarks></remarks>
        double M { get; set; }

        /// <summary>
        /// Gets or sets the coordinate value M.
        /// </summary>
        /// <value>The coordinate value M.</value>
        /// <remarks></remarks>
        ICoordinateM CoordinateValueM { get; set; }

        /// <summary>
        /// Equalses the M.
        /// </summary>
        /// <param name="other">The other.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool EqualsM(ICoordinateM other);

        /// <summary>
        /// Equals3s the DM.
        /// </summary>
        /// <param name="other">The other.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Equals3DM(ICoordinateM other);

    }
}
