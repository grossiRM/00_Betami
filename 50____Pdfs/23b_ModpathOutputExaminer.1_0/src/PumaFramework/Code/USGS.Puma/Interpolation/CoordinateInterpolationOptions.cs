using System;


namespace USGS.Puma.Interpolation
{
    /// <summary>
    /// Options for interpolating elevation and measure data associated with a coordinate point.
    /// </summary>
    public enum CoordinateInterpolationOptions
    {
        /// <summary>
        /// Interpolate elevation (Z value) only
        /// </summary>
        InterpolateElevation = 0,
        /// <summary>
        /// Interpolate measure (M value) only
        /// </summary>
        InterpolateMeasure = 1,
        /// <summary>
        /// Interpolate both elevation and measure values
        /// </summary>
        InterpolateElevationAndMeasure = 2
    }
}
