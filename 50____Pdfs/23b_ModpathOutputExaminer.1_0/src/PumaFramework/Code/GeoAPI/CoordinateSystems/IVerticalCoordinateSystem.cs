// Copyright 2005, 2006 - Morten Nielsen (www.iter.dk)
//
// This file is part of SharpMap.
// SharpMap is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
// 
// SharpMap is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License
// along with SharpMap; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA 

namespace GeoAPI.CoordinateSystems
{
    /// <summary>
    /// A one-dimensional coordinate system suitable for vertical measurements.
    /// </summary>
    /// <remarks></remarks>
	public interface IVerticalCoordinateSystem : ICoordinateSystem
	{
        /// <summary>
        /// Gets the vertical datum, which indicates the measurement method
        /// </summary>
        /// <value>The vertical datum.</value>
        /// <remarks></remarks>
		IVerticalDatum VerticalDatum { get; set; }

        /// <summary>
        /// Gets the units used along the vertical axis.
        /// </summary>
        /// <value>The vertical unit.</value>
        /// <remarks></remarks>
		ILinearUnit VerticalUnit { get; set; }
	}
}
