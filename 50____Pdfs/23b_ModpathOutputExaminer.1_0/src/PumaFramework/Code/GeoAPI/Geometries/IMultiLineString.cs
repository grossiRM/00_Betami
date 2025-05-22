namespace GeoAPI.Geometries
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public interface IMultiLineString : IMultiCurve
    {
        /// <summary>
        /// Reverses this instance.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        IMultiLineString Reverse();
    }
}
