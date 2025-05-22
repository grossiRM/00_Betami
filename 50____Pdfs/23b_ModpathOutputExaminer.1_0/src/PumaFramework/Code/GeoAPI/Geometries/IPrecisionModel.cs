using System;

namespace GeoAPI.Geometries
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public enum PrecisionModels
    {
        /// <summary> 
        /// Floating precision corresponds to the standard 
        /// double-precision floating-point representation, which is
        /// based on the IEEE-754 standard
        /// </summary>
        Floating = 0,

        /// <summary>
        /// Floating single precision corresponds to the standard
        /// single-precision floating-point representation, which is
        /// based on the IEEE-754 standard
        /// </summary>
        FloatingSingle = 1,

        /// <summary> 
        /// Fixed Precision indicates that coordinates have a fixed number of decimal places.
        /// The number of decimal places is determined by the log10 of the scale factor.
        /// </summary>
        Fixed = 2,
    }

    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public interface IPrecisionModel : IComparable, IComparable<IPrecisionModel>, IEquatable<IPrecisionModel>
    {
        /// <summary>
        /// Gets the type of the precision model.
        /// </summary>
        /// <remarks></remarks>
        PrecisionModels PrecisionModelType { get; }
        /// <summary>
        /// Gets a value indicating whether this instance is floating.
        /// </summary>
        /// <remarks></remarks>
        bool IsFloating { get; }
        /// <summary>
        /// Gets the maximum significant digits.
        /// </summary>
        /// <remarks></remarks>
        int MaximumSignificantDigits { get; }
        /// <summary>
        /// Gets the scale.
        /// </summary>
        /// <remarks></remarks>
        double Scale { get; }

        /// <summary>
        /// Makes the precise.
        /// </summary>
        /// <param name="val">The val.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        double MakePrecise(double val);
        /// <summary>
        /// Makes the precise.
        /// </summary>
        /// <param name="coord">The coord.</param>
        /// <remarks></remarks>
        void MakePrecise(ICoordinate coord);
    }
}
