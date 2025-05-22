using System;

namespace GeoAPI.Geometries
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public interface IEnvelope : ICloneable, IComparable, IComparable<IEnvelope>, IEquatable<IEnvelope>
    {
        /// <summary>
        /// Gets the area.
        /// </summary>
        /// <remarks></remarks>
        double Area { get; }

        /// <summary>
        /// Gets the width.
        /// </summary>
        /// <remarks></remarks>
        double Width { get; }

        /// <summary>
        /// Gets the height.
        /// </summary>
        /// <remarks></remarks>
        double Height { get; }

        /// <summary>
        /// Gets the max X.
        /// </summary>
        /// <remarks></remarks>
        double MaxX { get; }

        /// <summary>
        /// Gets the max Y.
        /// </summary>
        /// <remarks></remarks>
        double MaxY { get; }

        /// <summary>
        /// Gets the min X.
        /// </summary>
        /// <remarks></remarks>
        double MinX { get; }

        /// <summary>
        /// Gets the min Y.
        /// </summary>
        /// <remarks></remarks>
        double MinY { get; }

        /// <summary>
        /// Gets the centre.
        /// </summary>
        /// <remarks></remarks>
        ICoordinate Centre { get; }

        /// <summary>
        /// Determines whether [contains] [the specified x].
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns><c>true</c> if [contains] [the specified x]; otherwise, <c>false</c>.</returns>
        /// <remarks></remarks>
        bool Contains(double x, double y);

        /// <summary>
        /// Determines whether [contains] [the specified p].
        /// </summary>
        /// <param name="p">The p.</param>
        /// <returns><c>true</c> if [contains] [the specified p]; otherwise, <c>false</c>.</returns>
        /// <remarks></remarks>
        bool Contains(ICoordinate p);

        /// <summary>
        /// Determines whether [contains] [the specified other].
        /// </summary>
        /// <param name="other">The other.</param>
        /// <returns><c>true</c> if [contains] [the specified other]; otherwise, <c>false</c>.</returns>
        /// <remarks></remarks>
        bool Contains(IEnvelope other);

        /// <summary>
        /// Distances the specified env.
        /// </summary>
        /// <param name="env">The env.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        double Distance(IEnvelope env);

        /// <summary>
        /// Expands the by.
        /// </summary>
        /// <param name="distance">The distance.</param>
        /// <remarks></remarks>
        void ExpandBy(double distance);

        /// <summary>
        /// Expands the by.
        /// </summary>
        /// <param name="deltaX">The delta X.</param>
        /// <param name="deltaY">The delta Y.</param>
        /// <remarks></remarks>
        void ExpandBy(double deltaX, double deltaY);

        /// <summary>
        /// Expands to include.
        /// </summary>
        /// <param name="p">The p.</param>
        /// <remarks></remarks>
        void ExpandToInclude(ICoordinate p);

        /// <summary>
        /// Expands to include.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <remarks></remarks>
        void ExpandToInclude(double x, double y);

        /// <summary>
        /// Expands to include.
        /// </summary>
        /// <param name="other">The other.</param>
        /// <remarks></remarks>
        void ExpandToInclude(IEnvelope other);

        /// <summary>
        /// Inits this instance.
        /// </summary>
        /// <remarks></remarks>
        void Init();

        /// <summary>
        /// Inits the specified p.
        /// </summary>
        /// <param name="p">The p.</param>
        /// <remarks></remarks>
        void Init(ICoordinate p);

        /// <summary>
        /// Inits the specified env.
        /// </summary>
        /// <param name="env">The env.</param>
        /// <remarks></remarks>
        void Init(IEnvelope env);

        /// <summary>
        /// Inits the specified p1.
        /// </summary>
        /// <param name="p1">The p1.</param>
        /// <param name="p2">The p2.</param>
        /// <remarks></remarks>
        void Init(ICoordinate p1, ICoordinate p2);

        /// <summary>
        /// Inits the specified x1.
        /// </summary>
        /// <param name="x1">The x1.</param>
        /// <param name="x2">The x2.</param>
        /// <param name="y1">The y1.</param>
        /// <param name="y2">The y2.</param>
        /// <remarks></remarks>
        void Init(double x1, double x2, double y1, double y2);

        /// <summary>
        /// Intersections the specified env.
        /// </summary>
        /// <param name="env">The env.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IEnvelope Intersection(IEnvelope env);

        /// <summary>
        /// Translates the specified trans X.
        /// </summary>
        /// <param name="transX">The trans X.</param>
        /// <param name="transY">The trans Y.</param>
        /// <remarks></remarks>
        void Translate(double transX, double transY);

        /// <summary>
        /// Unions the specified point.
        /// </summary>
        /// <param name="point">The point.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IEnvelope Union(IPoint point);

        /// <summary>
        /// Unions the specified coord.
        /// </summary>
        /// <param name="coord">The coord.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IEnvelope Union(ICoordinate coord);

        /// <summary>
        /// Unions the specified box.
        /// </summary>
        /// <param name="box">The box.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        IEnvelope Union(IEnvelope box);

        /// <summary>
        /// Intersectses the specified p.
        /// </summary>
        /// <param name="p">The p.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Intersects(ICoordinate p);

        /// <summary>
        /// Intersectses the specified x.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Intersects(double x, double y);

        /// <summary>
        /// Intersectses the specified other.
        /// </summary>
        /// <param name="other">The other.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Intersects(IEnvelope other);

        /// <summary>
        /// Gets a value indicating whether this instance is null.
        /// </summary>
        /// <remarks></remarks>
        bool IsNull { get; }

        /// <summary>
        /// Sets to null.
        /// </summary>
        /// <remarks></remarks>
        void SetToNull();

        /// <summary>
        /// Zooms the specified per cent.
        /// </summary>
        /// <param name="perCent">The per cent.</param>
        /// <remarks></remarks>
        void Zoom(double perCent);

        /// <summary>
        /// Overlapses the specified other.
        /// </summary>
        /// <param name="other">The other.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Overlaps(IEnvelope other);

        /// <summary>
        /// Overlapses the specified p.
        /// </summary>
        /// <param name="p">The p.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Overlaps(ICoordinate p);

        /// <summary>
        /// Overlapses the specified x.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool Overlaps(double x, double y);

        /// <summary>
        /// Sets the centre.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <remarks></remarks>
        void SetCentre(double width, double height);

        /// <summary>
        /// Sets the centre.
        /// </summary>
        /// <param name="centre">The centre.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <remarks></remarks>
        void SetCentre(IPoint centre, double width, double height);

        /// <summary>
        /// Sets the centre.
        /// </summary>
        /// <param name="centre">The centre.</param>
        /// <remarks></remarks>
        void SetCentre(ICoordinate centre);

        /// <summary>
        /// Sets the centre.
        /// </summary>
        /// <param name="centre">The centre.</param>
        /// <remarks></remarks>
        void SetCentre(IPoint centre);

        /// <summary>
        /// Sets the centre.
        /// </summary>
        /// <param name="centre">The centre.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <remarks></remarks>
        void SetCentre(ICoordinate centre, double width, double height);                
    }
}
