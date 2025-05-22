//  FDGridUtil
using System;
using System.Collections;
using System.Data;
using System.Diagnostics;

namespace USGS.Puma.Core
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public interface ICellCenteredArealGrid
    {
        /// <summary>
        /// Gets the row count.
        /// </summary>
        /// <remarks></remarks>
        int RowCount { get; }
        /// <summary>
        /// Gets the column count.
        /// </summary>
        /// <remarks></remarks>
        int ColumnCount { get; }
        /// <summary>
        /// Gets or sets the angle.
        /// </summary>
        /// <value>The angle.</value>
        /// <remarks></remarks>
        double Angle { get; set; }
        /// <summary>
        /// Gets or sets the origin X.
        /// </summary>
        /// <value>The origin X.</value>
        /// <remarks></remarks>
        double OriginX { get; set; }
        /// <summary>
        /// Gets or sets the origin Y.
        /// </summary>
        /// <value>The origin Y.</value>
        /// <remarks></remarks>
        double OriginY { get; set; }
        /// <summary>
        /// Gets the row spacing.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        double GetRowSpacing(int index);
        /// <summary>
        /// Gets the column spacing.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        double GetColumnSpacing(int index);
        /// <summary>
        /// Gets the total height of the row.
        /// </summary>
        /// <remarks></remarks>
        double TotalRowHeight { get; }
        /// <summary>
        /// Gets the total width of the column.
        /// </summary>
        /// <remarks></remarks>
        double TotalColumnWidth { get; }
    }
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public interface ISerializeXml
    {
        /// <summary>
        /// Loads from XML.
        /// </summary>
        /// <param name="xmlString">The XML string.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        bool LoadFromXml(string xmlString);
        /// <summary>
        /// Saves as XML.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        string SaveAsXml();
        /// <summary>
        /// Saves as XML.
        /// </summary>
        /// <param name="elementName">Name of the element.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        string SaveAsXml(string elementName);
    }
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public interface IDataObject : ISerializeXml
    {
        /// <summary>
        /// Gets the type of the puma.
        /// </summary>
        /// <remarks></remarks>
        string PumaType { get; }
        /// <summary>
        /// Gets the default name.
        /// </summary>
        /// <remarks></remarks>
        string DefaultName { get; }
        /// <summary>
        /// Gets the version.
        /// </summary>
        /// <remarks></remarks>
        int Version { get; }
        /// <summary>
        /// Gets a value indicating whether this instance is valid.
        /// </summary>
        /// <remarks></remarks>
        bool IsValid { get; }
    }
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public interface IGridCell
    {
        /// <summary>
        /// Gets or sets the grid.
        /// </summary>
        /// <value>The grid.</value>
        /// <remarks></remarks>
        int Grid { get; set; }
        /// <summary>
        /// Gets or sets the layer.
        /// </summary>
        /// <value>The layer.</value>
        /// <remarks></remarks>
        int Layer { get; set; }
        /// <summary>
        /// Gets or sets the row.
        /// </summary>
        /// <value>The row.</value>
        /// <remarks></remarks>
        int Row { get; set; }
        /// <summary>
        /// Gets or sets the column.
        /// </summary>
        /// <value>The column.</value>
        /// <remarks></remarks>
        int Column { get; set; }
    }

} 
