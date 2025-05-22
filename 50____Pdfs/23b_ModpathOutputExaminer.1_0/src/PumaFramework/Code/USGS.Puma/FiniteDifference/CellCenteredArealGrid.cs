using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma;
using USGS.Puma.Core;
using USGS.Puma.Utilities;
using USGS.Puma.IO;
using USGS.Puma.NTS.Geometries;
using GeoAPI.Geometries;
using System.Xml;

namespace USGS.Puma.FiniteDifference
{
    /// <summary>
    /// This class represents a 2-dimensional areal finite-difference grid. It
    /// provides properties and methods for managing spatial data associated
    /// with the grid.
    /// </summary>
    /// <remarks>This class supports georeference coordinates through its Orign and Angle properties.
    /// In addition, it provides a number of methods that supply grid coordinates defining
    /// features related to the grid.</remarks>
    public class CellCenteredArealGrid : ICellCenteredArealGrid, IDataObject
    {
        #region Static Methods
        /// <summary>
        /// Creates a new instance of the class from XML data.
        /// </summary>
        /// <param name="xmlNode">The XML node.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public static CellCenteredArealGrid FromXML(XmlNode xmlNode)
        {
            try
            {
                CellCenteredArealGrid grid = new CellCenteredArealGrid();
                
                double result = 0;
                double[] spacing = null;
                System.Xml.XmlNode node;
                System.Xml.XmlNode root = xmlNode;

                if (root == null)
                { throw new Exception("Error loading Puma object from XML."); }

                if (!DataObjectUtility.IsValidPumaType(root, grid.PumaType))
                { throw new Exception("Error loading Puma object from XML. Incorrect PumaType."); }

                // Declare a list object to hold the grid spacing data
                IndexRangeValueList list = null;

                // Process rows
                node = root.SelectSingleNode("RowCount");
                int rowCount = int.Parse(node.InnerText);
                node = root.SelectSingleNode("RowSpacing");
                if (double.TryParse(node.InnerText, out result))
                { grid.RowSpacing.SetSize(rowCount, result); }
                else
                {
                    spacing = NumberArrayIO<double>.StringToArray(node.InnerText);
                    if (spacing.Length != rowCount)
                    { throw (new Exception("Error computing row spacing from XML.")); }
                    grid.RowSpacing.SetSize(spacing);
                }

                // Process columns
                node = root.SelectSingleNode("ColumnCount");
                int columnCount = int.Parse(node.InnerText);
                node = root.SelectSingleNode("ColumnSpacing");
                if (double.TryParse(node.InnerText, out result))
                { grid.ColumnSpacing.SetSize(columnCount, result); }
                else
                {
                    spacing = NumberArrayIO<double>.StringToArray(node.InnerText);
                    if (spacing.Length != columnCount)
                    { throw (new Exception("Error computing column spacing from XML.")); }
                    grid.ColumnSpacing.SetSize(spacing);
                }

                // Process Angle
                node = root.SelectSingleNode("Angle");
                grid.Angle = double.Parse(node.InnerText);

                // Process Origin
                node = root.SelectSingleNode("OriginX");
                grid.OriginX = double.Parse(node.InnerText);
                node = root.SelectSingleNode("OriginY");
                grid.OriginY = double.Parse(node.InnerText);

                // Process Projection
                grid.Projection = "";
                node = root.SelectSingleNode("Projection");
                if (node != null) grid.Projection = node.InnerText;

                return grid;

            }
            catch (Exception)
            {
                return null;
            }

        }
        /// <summary>
        /// Creates a new instance of the class from XML data.
        /// </summary>
        /// <param name="xmlData">The XML data.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public static CellCenteredArealGrid FromXML(string xmlData)
        {
            XmlDocument doc = new XmlDocument();
            doc.LoadXml(xmlData);
            XmlNode node = doc.DocumentElement as XmlNode;
            return FromXML(node);
        }

        /// <summary>
        /// Creates an XmlNode containing the description of the grid.
        /// </summary>
        /// <param name="grid">The grid.</param>
        /// <returns>An XmlNode object</returns>
        /// <remarks></remarks>
        public static XmlNode ToXML(CellCenteredArealGrid grid)
        {
            return ToXML(grid, grid.DefaultName);
        }
        /// <summary>
        /// Creates an XmlNode containing the description of the grid.
        /// </summary>
        /// <param name="grid">The grid.</param>
        /// <param name="elementName">Name of the element.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public static XmlNode ToXML(CellCenteredArealGrid grid, string elementName)
        {
            try
            {
                double[] spacing = null;
                string sXml = null;
                System.Xml.XmlNode node = null;
                System.Xml.XmlDocumentFragment docFrag = null;
                System.Xml.XmlDocument doc = DataObjectUtility.XmlWrapperDoc(grid, elementName);
                System.Xml.XmlNode root = doc.DocumentElement;

                // Append puma type
                DataObjectUtility.AppendPumaTypeAttribute(grid, root);

                // Append the row count
                node = root.AppendChild(doc.CreateElement("RowCount"));
                node.InnerText = grid.RowCount.ToString();

                // Append the column count
                node = root.AppendChild(doc.CreateElement("ColumnCount"));
                node.InnerText = grid.ColumnCount.ToString();

                // Append the rotation angle.
                node = root.AppendChild(doc.CreateElement("Angle"));
                node.InnerText = grid.Angle.ToString();

                // Write the origin point as xml and append it to the root element
                node = root.AppendChild(doc.CreateElement("OriginX"));
                node.InnerText = grid.OriginX.ToString();
                node = root.AppendChild(doc.CreateElement("OriginY"));
                node.InnerText = grid.OriginY.ToString();

                // Append row spacing
                root.AppendChild(doc.CreateElement("RowSpacing"));
                if (grid.RowSpacing.IsConstantSpacing)
                { root.LastChild.InnerText = grid.RowSpacing[1].ToString(); }
                else
                {
                    spacing = new double[grid.RowCount];
                    for (int i = 0; i < grid.RowCount; i++)
                    { spacing[i] = grid.RowSpacing[i + 1]; }
                    root.LastChild.InnerText = NumberArrayIO<double>.ArrayToString(spacing, true);
                }

                // Append column spacing
                root.AppendChild(doc.CreateElement("ColumnSpacing"));
                if (grid.ColumnSpacing.IsConstantSpacing)
                { root.LastChild.InnerText = grid.ColumnSpacing[1].ToString(); }
                else
                {
                    spacing = new double[grid.ColumnCount];
                    for (int i = 0; i < grid.ColumnCount; i++)
                    { spacing[i] = grid.ColumnSpacing[i + 1]; }
                    root.LastChild.InnerText = NumberArrayIO<double>.ArrayToString(spacing, true);
                }

                // Append projection
                root.AppendChild(doc.CreateElement("Projection"));
                root.LastChild.InnerText = grid.Projection;

                // Return the result as an xml string.
                return root;

            }
            catch (Exception)
            {
                throw new Exception("Error saving Puma object as XML.");
            }

        }

        /// <summary>
        /// Writes the description of the grid to an XML file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="grid">The grid.</param>
        /// <remarks></remarks>
        public static void Write(string filename, CellCenteredArealGrid grid)
        {
            Write(filename, grid, grid.DefaultName);
        }
        /// <summary>
        /// Writes the description of the grid to an XML file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="grid">The grid.</param>
        /// <param name="elementName">Name of the element.</param>
        /// <remarks></remarks>
        public static void Write(string filename,CellCenteredArealGrid grid, string elementName)
        {
            using (XmlTextWriter writer = new XmlTextWriter(filename, null))
            {
                writer.Formatting = Formatting.Indented;
                writer.Indentation = 2;
                Write(writer, grid, elementName); 
            }
        }
        /// <summary>
        /// Writes the description of the grid to an XML file.
        /// </summary>
        /// <param name="writer">The writer.</param>
        /// <param name="grid">The grid.</param>
        /// <remarks></remarks>
        public static void Write(XmlTextWriter writer, CellCenteredArealGrid grid)
        {
            Write(writer, grid, grid.DefaultName);
        }
        /// <summary>
        /// Writes the description of the grid to an XML file.
        /// </summary>
        /// <param name="writer">The writer.</param>
        /// <param name="grid">The grid.</param>
        /// <param name="elementName">Name of the element.</param>
        /// <remarks></remarks>
        public static void Write(XmlTextWriter writer, CellCenteredArealGrid grid, string elementName)
        {
            try
            {
                // Create an XmlNode and write it to the XmlTextWriter.
                XmlNode node = CellCenteredArealGrid.ToXML(grid, elementName);
                node.WriteTo(writer);
            }
            catch (Exception)
            {
                throw new Exception("Error saving Puma object as XML.");
            }



        }

        /// <summary>
        /// Reads an XML file and creates a new instance of CellCenteredArealGrid
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public static CellCenteredArealGrid Read(string filename)
        {
            XmlDocument doc = new XmlDocument();
            doc.Load(filename);
            XmlNode node = doc.DocumentElement as XmlNode;
            return FromXML(node);
        }
        /// <summary>
        /// Reads an XML file and creates a new instance of CellCenteredArealGrid
        /// </summary>
        /// <param name="reader">The reader.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public static CellCenteredArealGrid Read(XmlTextReader reader)
        {
            if (!reader.HasAttributes)
                return null;

            string sXml = reader.ReadOuterXml();
            return FromXML(sXml);
        }
        #endregion

        #region ArealGrid
        #region Fields
        /// <summary>
        /// 
        /// </summary>
        private double _AngleCos;
        /// <summary>
        /// 
        /// </summary>
        private double _AngleSin;
        /// <summary>
        /// 
        /// </summary>
        private GridSpacing _Rows;
        /// <summary>
        /// 
        /// </summary>
        private GridSpacing _Columns;
        
        #endregion

        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="T:System.Object"/> class.
        /// </summary>
        /// <remarks></remarks>
        public CellCenteredArealGrid()
        {  InitializeSquareCellGrid(1, 1, 1, (ICoordinate)(new Point(0, 0)), 0);  }
        /// <summary>
        /// Initializes a new instance of the <see cref="CellCenteredArealGrid"/> class from an XML data source.
        /// </summary>
        /// <param name="xmlString">The XML string.</param>
        /// <remarks></remarks>
        public CellCenteredArealGrid(string xmlString)
        {
            try
            {
                InitializeSquareCellGrid(1, 1, 1, (ICoordinate)(new Point(0, 0)), 0);
                if (!LoadFromXml(xmlString))
                { throw new Exception("Error creating ArealGrid object from XML."); }
                IsValid = true;
            }
            catch (Exception)
            {
                IsValid = false;
                throw new Exception("Error creating ArealGrid object from XML.");
            }
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="CellCenteredArealGrid"/> class.
        /// </summary>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <param name="cellSize">Size of the cell.</param>
        /// <param name="origin">The origin.</param>
        /// <param name="rotationAngle">The rotation angle.</param>
        /// <remarks></remarks>
        public CellCenteredArealGrid(int rows, int columns, double cellSize, ICoordinate origin, double rotationAngle)
        { InitializeSquareCellGrid(rows, columns, cellSize, origin, rotationAngle); }
        /// <summary>
        /// Initializes a new instance of the <see cref="CellCenteredArealGrid"/> class.
        /// </summary>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <param name="cellSize">Size of the cell.</param>
        /// <param name="originX">The origin X.</param>
        /// <param name="originY">The origin Y.</param>
        /// <param name="rotationAngle">The rotation angle.</param>
        /// <remarks></remarks>
        public CellCenteredArealGrid(int rows, int columns, double cellSize, double originX,double originY, double rotationAngle)
        { InitializeSquareCellGrid(rows, columns, cellSize, originX, originY, rotationAngle); }
        /// <summary>
        /// Initializes a new instance of the <see cref="CellCenteredArealGrid"/> class.
        /// </summary>
        /// <param name="rowSpacing">The row spacing.</param>
        /// <param name="columnSpacing">The column spacing.</param>
        /// <remarks></remarks>
        public CellCenteredArealGrid(Array1d<double> rowSpacing, Array1d<double> columnSpacing)
        {
            try
            {
                Angle = 0;
                _Origin = (ICoordinate)(new Coordinate());
                _Rows = new GridSpacing();
                _Rows.SetSize(rowSpacing.GetValues());
                _Columns = new GridSpacing();
                _Columns.SetSize(columnSpacing.GetValues());

                IsValid = true;

            }
            catch (Exception)
            {
                IsValid = false;
                throw new Exception("Error creating ArealGrid object.");
            }
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="CellCenteredArealGrid"/> class.
        /// </summary>
        /// <param name="rowSpacing">The row spacing.</param>
        /// <param name="columnSpacing">The column spacing.</param>
        /// <param name="originX">The origin X.</param>
        /// <param name="originY">The origin Y.</param>
        /// <param name="rotationAngle">The rotation angle.</param>
        /// <remarks></remarks>
        public CellCenteredArealGrid(Array1d<double> rowSpacing, Array1d<double> columnSpacing, double originX, double originY, double rotationAngle)
            : this(rowSpacing, columnSpacing)
        {
            try
            {
                Angle = rotationAngle;
                OriginX = originX;
                OriginY = originY;
                IsValid = true;
            }
            catch (Exception)
            {
                IsValid = false;
                throw new Exception("Error creating ArealGrid object.");
            }
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="CellCenteredArealGrid"/> class.
        /// </summary>
        /// <param name="rowSpacing">The row spacing.</param>
        /// <param name="columnSpacing">The column spacing.</param>
        /// <remarks></remarks>
        public CellCenteredArealGrid(Array1d<float> rowSpacing, Array1d<float> columnSpacing) : this(rowSpacing.GetCopyAsDouble(), columnSpacing.GetCopyAsDouble()) { }
        /// <summary>
        /// Initializes a new instance of the <see cref="CellCenteredArealGrid"/> class.
        /// </summary>
        /// <param name="rowSpacing">The row spacing.</param>
        /// <param name="columnSpacing">The column spacing.</param>
        /// <param name="originX">The origin X.</param>
        /// <param name="originY">The origin Y.</param>
        /// <param name="rotationAngle">The rotation angle.</param>
        /// <remarks></remarks>
        public CellCenteredArealGrid(Array1d<float> rowSpacing, Array1d<float> columnSpacing, float originX, float originY, float rotationAngle)
            : this(rowSpacing.GetCopyAsDouble(), columnSpacing.GetCopyAsDouble(), Convert.ToDouble(originX), Convert.ToDouble(originY), Convert.ToDouble(rotationAngle)) { }
        /// <summary>
        /// Initializes a new instance of the <see cref="CellCenteredArealGrid"/> class.
        /// </summary>
        /// <param name="origin">The origin.</param>
        /// <param name="totalColumnWidth">Total width of the column.</param>
        /// <param name="totalRowHeight">Total height of the row.</param>
        /// <param name="rotationAngle">The rotation angle.</param>
        /// <param name="cellSize">Size of the cell.</param>
        /// <remarks></remarks>
        public CellCenteredArealGrid(ICoordinate origin, double totalColumnWidth, double totalRowHeight, double rotationAngle, double cellSize)
        {
            try
            {
                if ((totalColumnWidth > 0) & (totalRowHeight > 0))
                {
                    InitializeSquareCellGrid(0, 0, cellSize, origin, rotationAngle);
                    RowSpacing.SetSize(cellSize, totalRowHeight);
                    ColumnSpacing.SetSize(cellSize, totalColumnWidth);
                    IsValid = true;
                }
                else
                {
                    IsValid = false;
                    throw new Exception("The width and height of the grid must be greater than 0.");
                }
            }
            catch (Exception)
            {
                IsValid = false;
                throw new Exception("An error occurred creating the ArealGrid object."); 
            } 
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="CellCenteredArealGrid"/> class.
        /// </summary>
        /// <param name="origin">The origin.</param>
        /// <param name="totalColumnWidth">Total width of the column.</param>
        /// <param name="totalRowHeight">Total height of the row.</param>
        /// <param name="rotationAngle">The rotation angle.</param>
        /// <param name="maxCells">The max cells.</param>
        /// <param name="cellSizePrecision">The cell size precision.</param>
        /// <remarks></remarks>
        public CellCenteredArealGrid(ICoordinate origin, double totalColumnWidth, double totalRowHeight, double rotationAngle, int maxCells, int cellSizePrecision)
        {
            double width = 0;
            double height = 0;
            int columns = 0;
            int rows = 0;
            double cellSize = 0;

            try
            {
                if ((totalColumnWidth > 0) & (totalRowHeight > 0))
                {
                    if (totalColumnWidth >= totalRowHeight)
                    {
                        columns = maxCells;
                        cellSize = Math.Round(totalColumnWidth / columns, cellSizePrecision);
                        rows = (int)Math.Round(totalRowHeight / cellSize, 0);
                        if (rows < 1)
                        { rows = 1; }
                    }
                    else
                    {
                        rows = maxCells;
                        cellSize = Math.Round(totalRowHeight / rows, cellSizePrecision);
                        columns = (int)Math.Round(totalColumnWidth / cellSize, 0);
                        if (columns < 1)
                        { columns = 1; }
                    }

                    InitializeSquareCellGrid(rows, columns, cellSize, origin, rotationAngle);

                    IsValid = true;

                }
                else
                {
                    IsValid = false;
                    throw new Exception("Grid width and height must be greater than 0."); 
                }
            }
            catch (Exception ex)
            {
                IsValid = false;
                throw new Exception("Could not create the areal grid."); 
            } 
        }
        #endregion

        #region Public Members

        #region Primary Public Members
        /// <summary>
        /// Gets a value indicating whether this instance is a square cell grid.
        /// </summary>
        /// <value><c>true</c> if this instance is a square cell grid; otherwise, <c>false</c>.</value>
        /// <remarks></remarks>
        public bool IsSquareCellGrid
        {
            get
            {
                if (!_Rows.IsConstantSpacing) return false;
                if (!_Columns.IsConstantSpacing) return false;
                if (_Rows[1] == _Columns[1]) return true;
                return false;
            }
        }
        /// <summary>
        /// Gets or sets the number of rows.
        /// </summary>
        /// <value>The number of rows in the grid.</value>
        /// <remarks></remarks>
        public int RowCount
        {
            get
            { return RowSpacing.Count; }
            set
            {
                if (RowSpacing.Count < 1) return;
                if (value == RowSpacing.Count) return;

                bool addRows = false;
                if (value > RowSpacing.Count) addRows = true;

                do
                {
                    if (value == RowSpacing.Count) return;
                    if (addRows)
                    { RowSpacing.Add(RowSpacing[RowCount]); }
                    else
                    { RowSpacing.Remove(RowSpacing.Count); }
                } while (true);
            }
        }
        /// <summary>
        /// Gets or sets the number of columns.
        /// </summary>
        /// <value>The number of columns in the grid.</value>
        /// <remarks></remarks>
        public int ColumnCount
        {
            get
            { return ColumnSpacing.Count; }
            set
            {
                if (ColumnSpacing.Count < 1) return;
                if (value == ColumnSpacing.Count) return;

                bool addColumns = false;
                if (value > ColumnSpacing.Count) addColumns = true;

                do
                {
                    if (value == ColumnSpacing.Count) return;
                    if (addColumns)
                    { ColumnSpacing.Add(ColumnSpacing[ColumnCount]); }
                    else
                    { ColumnSpacing.Remove(ColumnSpacing.Count); }
                } while (true);

            }
        }

        /// <summary>
        /// 
        /// </summary>
        private double _Angle = 0;
        /// <summary>
        /// Gets or sets the angle of rotation for the grid around its origin.
        /// </summary>
        /// <value>The angle.</value>
        /// <remarks>
        /// The angle is measured relative to the positive x-axis direction. 
        /// The value increases in the counter-clockwise direction of rotation.
        /// An angle equal to 0 implies a grid with its row and column directions
        /// aligned with the x and y coordinate axes.
        /// </remarks>
        public double Angle
        {
            get { return _Angle; }
            set
            {
                _Angle = NormalizeAngle(value);
                MathUtility.ComputeCosSin(_Angle, ref _AngleCos, ref _AngleSin, false);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        private ICoordinate _Origin;
        /// <summary>
        /// Gets or sets the origin of the grid.
        /// </summary>
        /// <value>The origin.</value>
        /// <remarks></remarks>
        public ICoordinate Origin
        {
            get
            { return _Origin; }
            set
            {
                // Don't allow the origin to be set to null.
                if (value == null)
                    throw new NullReferenceException();

                _Origin = value;

            }
        }

        /// <summary>
        /// Gets or sets the x coordinate of the grid origin.
        /// </summary>
        /// <value>The origin X.</value>
        /// <remarks></remarks>
        public double OriginX
        {
            get
            {
                if (_Origin == null) _Origin = new Coordinate() as ICoordinate;
                return _Origin.X;
            }
            set
            {
                if (_Origin == null) _Origin = new Coordinate() as ICoordinate;
                _Origin.X = value;

            }
        }

        /// <summary>
        /// Gets or sets the y coordinate of the grid origin.
        /// </summary>
        /// <value>The origin Y.</value>
        /// <remarks></remarks>
        public double OriginY
        {
            get
            {
                if (_Origin == null) _Origin = new Coordinate() as ICoordinate;
                return _Origin.Y;
            }
            set
            {
                if (_Origin == null) _Origin = new Coordinate() as ICoordinate;
                _Origin.Y=value;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        private string _Projection = "";
        /// <summary>
        /// Gets or sets the description of the coordinate system projection.
        /// </summary>
        /// <value>A text-based description of the projection information.</value>
        /// <remarks>The text string is not used directly by the class. However, it can be
        /// used to store a text-based description of the coordinate system projection
        /// in the same format used to represent projection data in ESRI shapefiles.</remarks>
        public string Projection
        {
            get { return _Projection; }
            set { _Projection = value; }
        }

        /// <summary>
        /// Gets the total number of cells in the grid.
        /// </summary>
        /// <remarks></remarks>
        public int CellCount
        { get { return RowCount * ColumnCount; } }

        /// <summary>
        /// Gets an array containing the row spacing for the grid.
        /// </summary>
        /// <returns>An array containing the height of each row.</returns>
        /// <remarks>The number of elements in the array is equal to the number of rows
        /// in the grid. The array is zero-based.</remarks>
        public Array1d<double> GetRowSpacing()
        {
            Array1d<double> a = new Array1d<double>(this.RowCount);
            for (int i = 1; i <= this.RowCount; i++)
            {
                a[i] = GetRowSpacing(i);
            }
            return a;
        }
        /// <summary>
        /// Gets the row height for the specified row index.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns>A value of type double.</returns>
        /// <remarks></remarks>
        public double GetRowSpacing(int index)
        { return _Rows[index]; }

        /// <summary>
        /// Gets an array containing the column spacing for the grid.
        /// </summary>
        /// <returns>An array containing the width of each column.</returns>
        /// <remarks>The number of elements in the array is equal to the number of columns
        /// in the grid. The array is zero-based.</remarks>
        public Array1d<double> GetColumnSpacing()
        {
            Array1d<double> a = new Array1d<double>(this.ColumnCount);
            for (int i = 1; i <= this.ColumnCount; i++)
            {
                a[i] = GetRowSpacing(i);
            }
            return a;
        }
        /// <summary>
        /// Gets the column width for the specified column index.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns>A value of type double.</returns>
        /// <remarks></remarks>
        public double GetColumnSpacing(int index)
        { return _Columns[index]; }

        /// <summary>
        /// Gets a value indicating if the row spacing is a constant value.
        /// </summary>
        /// <returns>true is the row spacing is constant. Otherwise, returns false.</returns>
        /// <remarks></remarks>
        public bool RowSpacingIsConstant()
        {
            return RowSpacing.IsConstantSpacing;
        }
        /// <summary>
        /// Gets a value indicating if the column spacing is a constant value.
        /// </summary>
        /// <returns>true is the row spacing is constant. Otherwise, returns false.</returns>
        /// <remarks></remarks>
        public bool ColumnSpacingIsConstant()
        {
            return ColumnSpacing.IsConstantSpacing;
        }
        /// <summary>
        /// Gets the total height of the grid as the sum of all the row heights.
        /// </summary>
        /// <remarks></remarks>
        public double TotalRowHeight
        {
            get
            { return _Rows.TotalLength; }
        }
        /// <summary>
        /// Gets the total column width of the grid as the sum of all the column widths.
        /// </summary>
        /// <remarks></remarks>
        public double TotalColumnWidth
        {
            get
            { return _Columns.TotalLength; }
        }

        /// <summary>
        /// Sets the row and column spacing for the entire grid.
        /// </summary>
        /// <param name="rowSpacing">An array containing the height of all rows in the grid.</param>
        /// <param name="columnSpacing">An array of type double containing the width of all columns in the grid.</param>
        /// <remarks></remarks>
        public void SetSizeAndSpacing(Array1d<double> rowSpacing, Array1d<double> columnSpacing)
        {
            try
            {
                _Rows.SetSize(rowSpacing.GetValues());
                _Columns.SetSize(columnSpacing.GetValues());
                IsValid = true;
            }
            catch (Exception)
            {
                IsValid = false;
                throw new Exception("Error setting grid dimensions and spacing.");
            }

        }
        /// <summary>
        /// Sets the row and column spacing for the entire grid using a constant spacing
        /// for all rows and columns.
        /// </summary>
        /// <param name="rows">The number of rows in the grid.</param>
        /// <param name="columns">The number of columns in the grid.</param>
        /// <param name="cellSize">The value of spacing used for all rows and columns.</param>
        /// <remarks></remarks>
        public void SetSizeAndSpacing(int rows, int columns, double cellSize)
        {
            _Rows.SetSize(rows, cellSize);
            _Columns.SetSize(columns, cellSize);
        }

        /// <summary>
        /// Sets the grid spacing for a square cell grid.
        /// </summary>
        /// <param name="cellSize">The value of spacing used for all rows and columns.</param>
        /// <remarks>This method assumes that the number of rows and columns has already been
        /// specified. Specifying the cell size in this method changes the overall
        /// size of the grid. The number of rows and columns are not changed.</remarks>
        public void SetSquareCellSize(double cellSize)
        {
            _Rows.SetSize(cellSize, _Rows.TotalLength);
            _Columns.SetSize(cellSize, _Columns.TotalLength);
        }
        /// <summary>
        /// Sets the grid spacing and approximate overall dimension of the grid.
        /// The cell size and total height and width are used to compute the
        /// number of rows and columns.
        /// </summary>
        /// <param name="cellSize">The value of spacing used for all rows and columns.</param>
        /// <param name="totalRowHeight">The approximate total height of the desired grid.</param>
        /// <param name="totalColumnWidth">The approximate total width of the desired grid.</param>
        /// <remarks>The values for totalRowHeight and totalColumnWidth are approximate.
        /// Those values are divided by the specified value of cellSize. The resulting
        /// values are then truncated to determine the integer values for the number
        /// of rows and columns. Consequently, the actual total height and width of
        /// the resulting grid may be slightly less that the values specified for
        /// totalRowHeight and totalColumnWidth.</remarks>
        public void SetSquareCellSize(double cellSize, double totalRowHeight, double totalColumnWidth)
        {
            _Rows.SetSize(cellSize,totalRowHeight);
            _Columns.SetSize(cellSize, totalColumnWidth);
        }

        /// <summary>
        /// Sets the row spacing for the specified row index
        /// </summary>
        /// <param name="index">An integer representing the row index.</param>
        /// <param name="spacing">A double value representing the spacing for the specified row.</param>
        /// <remarks></remarks>
        public void SetRowSpacing(int index, double spacing)
        { _Rows[index] = spacing; }
        /// <summary>
        /// Sets the row spacing and the number of rows in the grid.
        /// </summary>
        /// <param name="spacing">An array containing the height of all rows in the grid.</param>
        /// <remarks>The length of the specified array is used to set the number of rows in the grid.</remarks>
        public void SetRowSpacing(Array1d<double> spacing)
        { RowSpacing.SetSize(spacing.GetValues()); }

        /// <summary>
        /// Sets the column spacing and the number of columns in the grid.
        /// </summary>
        /// <param name="spacing">An array containing the width of all columns in the grid.</param>
        /// <remarks>The length of the specified array is used to set the number of columns in the grid.</remarks>
        public void SetColumnSpacing(Array1d<double> spacing)
        { ColumnSpacing.SetSize(spacing.GetValues()); }
        /// <summary>
        /// Sets the column spacing for the specified column index.
        /// </summary>
        /// <param name="index">An integer representing the column index.</param>
        /// <param name="spacing">A double value representing the width of the specified column.</param>
        /// <remarks></remarks>
        public void SetColumnSpacing(int index, double spacing)
        { _Columns[index] = spacing; }

        /// <summary>
        /// Adds a row to the grid.
        /// </summary>
        /// <param name="spacing">A value of type double representing the heigth of the row.</param>
        /// <remarks>The row is added after the last row in the grid.</remarks>
        public void AddRow(double spacing)
        { RowSpacing.Add(spacing); }
        /// <summary>
        /// Adds a number of rows to the grid.
        /// </summary>
        /// <param name="spacing">An array of containing the heigths of the rows to be added.</param>
        /// <remarks>The rows are added after the last row in the grid.</remarks>
        public void AddRow(Array1d<double> spacing)
        { RowSpacing.Add(spacing.GetValues()); }

        /// <summary>
        /// A new row is inserted at the specified row index.
        /// </summary>
        /// <param name="index">The index at which the row will be added.</param>
        /// <param name="spacing">The height of the row that will be added.</param>
        /// <remarks>The first row in the grid has an index value equal to 1.</remarks>
        public void InsertRow(int index, double spacing)
        { RowSpacing.Insert(index, spacing); }
        /// <summary>
        /// Inserts a number of rows into the grid beginning at the specified index value.
        /// </summary>
        /// <param name="index">The index at which the rows will be inserted.</param>
        /// <param name="spacing">An array representing the height of the rows.</param>
        /// <remarks>The values specified for index refer to the row numbers of the grid.
        /// The first row of the grid has an index value equal to 1. However, the spacing array
        /// is zero-based.</remarks>
        public void InsertRow(int index, Array1d<double> spacing)
        { RowSpacing.Insert(index, spacing.GetValues()); }

        /// <summary>
        /// Removes the row.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <remarks></remarks>
        public void RemoveRow(int index)
        { RowSpacing.Remove(index); }
        /// <summary>
        /// Removes the row.
        /// </summary>
        /// <param name="fromIndex">From index.</param>
        /// <param name="toIndex">To index.</param>
        /// <remarks></remarks>
        public void RemoveRow(int fromIndex, int toIndex)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        /// <summary>
        /// Add a column to the grid.
        /// </summary>
        /// <param name="spacing">The width of the column being added.</param>
        /// <remarks>The column is added after the last column in the grid.</remarks>
        public void AddColumn(double spacing)
        { ColumnSpacing.Add(spacing); }
        /// <summary>
        /// Add an array of columns to the grid.
        /// </summary>
        /// <param name="spacing">The widths of all the columns being added to the grid.</param>
        /// <remarks>The columns are added after the last column in the grid.</remarks>
        public void AddColumn(Array1d<double> spacing)
        { ColumnSpacing.Add(spacing.GetValues()); }

        /// <summary>
        /// Insert a column in the grid.
        /// </summary>
        /// <param name="index">The column index at which to insert the new column.</param>
        /// <param name="spacing">The width of the column being added.</param>
        /// <remarks></remarks>
        public void InsertColumn(int index, double spacing)
        { ColumnSpacing.Insert(index, spacing); }
        /// <summary>
        /// Insert an array of columns in the grid.
        /// </summary>
        /// <param name="index">The column index at which to insert the new columns.</param>
        /// <param name="spacing">An array containing the widths of the columns being added.</param>
        /// <remarks></remarks>
        public void InsertColumn(int index, Array1d<double> spacing)
        { ColumnSpacing.Insert(index, spacing.GetValues()); }

        /// <summary>
        /// Removes the column.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <remarks></remarks>
        public void RemoveColumn(int index)
        { ColumnSpacing.Remove(index); }
        /// <summary>
        /// Removes the column.
        /// </summary>
        /// <param name="fromIndex">From index.</param>
        /// <param name="toIndex">To index.</param>
        /// <remarks></remarks>
        public void RemoveColumn(int fromIndex, int toIndex)
        { throw new Exception("The method or operation is not implemented."); }

        /// <summary>
        /// Gets the node point.
        /// </summary>
        /// <param name="cell">The cell.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public ICoordinate GetNodePoint(GridCell cell)
        {
            try
            {
                double x;
                double y;
                if (TryGetNodePointRelativeToGrid(cell, out x, out y))
                {
                    TransformRelativeToGlobal(ref x, ref y);
                    return (ICoordinate)(new Coordinate(x, y));
                }
                else
                {
                    return null;
                }
            }
            catch (Exception)
            { return null; }
        }

        /// <summary>
        /// Gets the node point relative to grid.
        /// </summary>
        /// <param name="cell">The cell.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public ICoordinate GetNodePointRelativeToGrid(GridCell cell)
        {
            try
            {
                double x;
                double y;
                if (TryGetNodePointRelativeToGrid(cell, out x, out y))
                    return (ICoordinate)(new Coordinate(x, y));
                else
                    return null;
            }
            catch (Exception)
            { return null; }
        }

        /// <summary>
        /// Tries the get node point.
        /// </summary>
        /// <param name="cell">The cell.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public bool TryGetNodePoint(GridCell cell, out double x, out double y)
        {
            x = 0;
            y = 0;

            try
            {
                if (TryGetNodePointRelativeToGrid(cell, out x, out y))
                {
                    TransformRelativeToGlobal(ref x,ref y);
                    return true;
                }
                else
                {
                    x = 0;
                    y = 0;
                    return false;
                }
            }
            catch (Exception)
            {
                return false;
            }

        }

        /// <summary>
        /// Tries the get node point relative to grid.
        /// </summary>
        /// <param name="cell">The cell.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public bool TryGetNodePointRelativeToGrid(GridCell cell, out double x, out double y)
        {
            x = 0;
            y = 0;

            try
            {
                x = ColumnSpacing.GetCenterOffset(cell.Column);
                y = RowSpacing.TotalLength - RowSpacing.GetCenterOffset(cell.Row);
                return true;
            }
            catch (Exception)
            {
                return false;
            }

        }

        /// <summary>
        /// Transforms from relative.
        /// </summary>
        /// <param name="pt">The pt.</param>
        /// <remarks></remarks>
        public void TransformRelativeToGlobal(ICoordinate pt)
        {
            if (pt == null) return;
            double x = pt.X;
            double y = pt.Y;
            TransformRelativeToGlobal(ref x, ref y);
            pt.X = x;
            pt.Y = y;
        }
        /// <summary>
        /// Transforms a geometry with coordinates that are relative to the grid to
        /// the equivalent geometry with global coordinates.
        /// </summary>
        /// <param name="geom">The geometry to be transformed.</param>
        /// <remarks></remarks>
        public void TransformRelativeToGlobal(IGeometry geometry)
        {
            if (geometry == null) return;
            ICoordinate[] coord = geometry.Coordinates;
            foreach (ICoordinate c in coord)
            {
                TransformRelativeToGlobal(c);
            }
            geometry.GeometryChanged();
        }
        /// <summary>
        /// Transforms from relative.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <remarks></remarks>
        public void TransformRelativeToGlobal(ref double x, ref double y)
        {
            double rX = 0;
            double rY = 0;

            // Translate origin
            x += Origin.X;
            y += Origin.Y;

            // Rotate around origin
            if (Angle != 0)
            {
                MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
                x = rX;
                y = rY;
            }

        }

        /// <summary>
        /// Transforms to relative.
        /// </summary>
        /// <param name="pt">The pt.</param>
        /// <remarks></remarks>
        public void TransformGlobalToRelative(ICoordinate pt)
        {
            if (pt == null) return;
            double x = pt.X;
            double y = pt.Y;
            TransformGlobalToRelative(ref x, ref y);
            pt.X = x;
            pt.Y = y;
        }
        /// <summary>
        /// Transforms to relative.
        /// </summary>
        /// <param name="geometry">The geometry.</param>
        /// <remarks></remarks>
        public void TransformGlobalToRelative(IGeometry geometry)
        {
            if (geometry == null) return;
            ICoordinate[] coord = geometry.Coordinates;
            foreach (ICoordinate c in coord)
            {
                TransformGlobalToRelative(c);
            }
            geometry.GeometryChanged();
        }
        /// <summary>
        /// Transforms to relative.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <remarks></remarks>
        public void TransformGlobalToRelative(ref double x, ref double y)
        {
            double rX = 0.0;
            double rY = 0.0;
            if (Angle == 0.0)
            {
                rX = x;
                rY = y;
            }
            else
            {
                MathUtility.FastBackwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
            }

            rX -= Origin.X;
            rY -= Origin.Y;
            x = rX;
            y = rY;
        }

        /// <summary>
        /// Finds the row column.
        /// </summary>
        /// <param name="location">The location.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public GridCell FindRowColumn(ICoordinate location)
        {
            double rX = 0;
            double rY = 0;

            try
            {
                if (Angle == 0.0)
                {
                    rX = location.X;
                    rY = location.Y;
                }
                else
                {
                    MathUtility.FastBackwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, location.X, location.Y, ref rX, ref rY);
                }

                int column = _Columns.FindCellIndex(rX - Origin.X);
                if (column < 1) return null;
                double rOffset = _Rows.TotalLength - (rY - Origin.Y);
                int row = _Rows.FindCellIndex(rOffset);
                if (row < 1) return null;

                return new GridCell(row, column);

            }
            catch (Exception)
            {
                return null;
            }
            finally
            {

            }
        }

        /// <summary>
        /// Gets the row gridline.
        /// </summary>
        /// <param name="rowIndex">Index of the row.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public IMultiLineString GetRowGridline(int rowIndex)
        {
            return GetRowGridline(rowIndex, 1, ColumnCount);
        }
        /// <summary>
        /// Gets the row gridline.
        /// </summary>
        /// <param name="rowIndex">Index of the row.</param>
        /// <param name="fromColumn">From column.</param>
        /// <param name="toColumn">To column.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public IMultiLineString GetRowGridline(int rowIndex, int fromColumn, int toColumn)
        {
            try
            {
                double rX = 0;
                double rY = 0;
                double x;
                double y;
                ICoordinate p1 = null;
                ICoordinate p2 = null;

                y = Origin.Y + (RowSpacing.TotalLength - RowSpacing.GetEdgeOffset(rowIndex));

                // point 1
                x = Origin.X + ColumnSpacing.GetEdgeOffset(fromColumn - 1);
                if ((Angle == 0))
                { p1 = (ICoordinate)(new Coordinate(x, y)); }
                else
                {
                    MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
                    p1 = (ICoordinate)(new Coordinate(rX, rY));
                }

                // point 2
                x = Origin.X + ColumnSpacing.GetEdgeOffset(toColumn);
                if ((Angle == 0))
                { p2 = (ICoordinate)(new Coordinate(x, y)); }
                else
                {
                    MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
                    p2 = (ICoordinate)(new Coordinate(rX, rY));
                }
                
                ICoordinate[] coords = new ICoordinate[2] { p1, p2 };
                ILineString ls = new LineString(coords);
                ILineString[] lsArray = new ILineString[1] { ls };
                IMultiLineString mls = new MultiLineString(lsArray);

                return mls;

            }
            catch (Exception)
            { return null; }

        }

        /// <summary>
        /// Gets the column gridline.
        /// </summary>
        /// <param name="columnIndex">Index of the column.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public IMultiLineString GetColumnGridline(int columnIndex)
        {
            return GetColumnGridline(columnIndex, 1, RowCount);
        }
        /// <summary>
        /// Gets the column gridline.
        /// </summary>
        /// <param name="columnIndex">Index of the column.</param>
        /// <param name="fromRow">From row.</param>
        /// <param name="toRow">To row.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public IMultiLineString GetColumnGridline(int columnIndex, int fromRow, int toRow)
        {
            try
            {
                double rX = 0;
                double rY = 0;
                double x;
                double y;
                ICoordinate p1 = null;
                ICoordinate p2 = null;

                if (columnIndex < 0 || columnIndex > ColumnCount)
                    throw new IndexOutOfRangeException();

                if (fromRow < 1 || fromRow > RowCount)
                    throw new IndexOutOfRangeException();

                if (toRow < 1 || toRow > RowCount)
                    throw new IndexOutOfRangeException();

                if (fromRow > toRow)
                    throw new ArgumentException();

                // point 1
                x = Origin.X + ColumnSpacing.GetEdgeOffset(columnIndex);

                y = Origin.Y + (RowSpacing.TotalLength - RowSpacing.GetEdgeOffset(fromRow - 1));
                if ((Angle == 0))
                { p1 = (ICoordinate)(new Coordinate(x, y)); }
                else
                {
                    MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
                    p1 = (ICoordinate)(new Coordinate(rX, rY));
                }

                // point 2
                y = Origin.Y + (RowSpacing.TotalLength - RowSpacing.GetEdgeOffset(toRow));
                if ((Angle == 0))
                { p2 = (ICoordinate)(new Coordinate(x, y)); }
                else
                {
                    MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
                    p2 = (ICoordinate)(new Coordinate(rX, rY));
                }

                ICoordinate[] coords = new ICoordinate[2] { p1, p2 };
                ILineString ls = new LineString(coords);
                ILineString[] lsArray = new ILineString[1] { ls };
                IMultiLineString mls = new MultiLineString(lsArray);

                return mls;


            }
            catch (Exception ex)
            {
                throw;
            }

        }

        /// <summary>
        /// Gets the corner points.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public ICoordinate[] GetCornerPoints()
        {
            return GetCornerPoints(new GridCell(1,1),new GridCell(RowCount,ColumnCount));
        }
        /// <summary>
        /// Gets the corner points.
        /// </summary>
        /// <param name="cell">The cell.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public ICoordinate[] GetCornerPoints(GridCell cell)
        {
            return GetCornerPoints(cell,new GridCell(cell.Row,cell.Column));
        }
        /// <summary>
        /// Gets the corner points.
        /// </summary>
        /// <param name="upperLeft">The upper left.</param>
        /// <param name="lowerRight">The lower right.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public ICoordinate[] GetCornerPoints(GridCell upperLeft, GridCell lowerRight)
        {
            double[] elevation = new double[4] { 0.0, 0.0, 0.0, 0.0 };
            return GetCornerPoints(upperLeft, lowerRight, elevation);

            //try
            //{
            //    double rX = 0;
            //    double rY = 0;
            //    double x;
            //    double y;

            //    if (upperLeft == null)
            //        throw new NullReferenceException();
            //    if (lowerRight == null)
            //        throw new NullReferenceException();
            //    if (upperLeft.Row < 1 || upperLeft.Row > RowCount || upperLeft.Column < 1 || upperLeft.Column > ColumnCount)
            //        throw new ArgumentException("Invalid cell location.");
            //    if (lowerRight.Row < 1 || lowerRight.Row > RowCount || lowerRight.Column < 1 || lowerRight.Column > ColumnCount)
            //        throw new ArgumentException("Invalid cell location.");

            //    List<ICoordinate> list = new List<ICoordinate>();

            //    // point 1
            //    x = Origin.X;
            //    if (upperLeft.Column > 1)
            //    { x += ColumnSpacing.GetEdgeOffset(upperLeft.Column - 1); }
            //    y = Origin.Y +  (TotalRowHeight - RowSpacing.GetEdgeOffset(lowerRight.Row));
            //    if ((Angle == 0))
            //    {
            //        rX = x;
            //        rY = y;
            //    }
            //    else
            //    {
            //        MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
            //    }
            //    list.Add( (ICoordinate)(new Coordinate(rX, rY)) );

            //    // point 2
            //    if (upperLeft.Row == 1)
            //    { y = Origin.Y + TotalRowHeight; }
            //    else
            //    { y = Origin.Y + (TotalRowHeight - RowSpacing.GetEdgeOffset(upperLeft.Row - 1)); }
            //    if ((Angle == 0))
            //    {
            //        rX = x;
            //        rY = y;
            //    }
            //    else
            //    {
            //        MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
            //    }
            //    list.Add((ICoordinate)(new Coordinate(rX, rY)));
                
            //    // point 3
            //    x = Origin.X + ColumnSpacing.GetEdgeOffset(lowerRight.Column);
            //    if ((Angle == 0))
            //    {
            //        rX = x;
            //        rY = y;
            //    }
            //    else
            //    {
            //        MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
            //    }
            //    list.Add((ICoordinate)(new Coordinate(rX, rY)));
                
            //    // point 4
            //    y = Origin.Y + (TotalRowHeight - RowSpacing.GetEdgeOffset(lowerRight.Row));
            //    if ((Angle == 0))
            //    {
            //        rX = x;
            //        rY = y;
            //    }
            //    else
            //    {
            //        MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
            //    }
            //    list.Add((ICoordinate)(new Coordinate(rX, rY)));

            //    ICoordinate[] ptList = list.ToArray();
            //    return ptList;
            //}
            //catch (Exception ex)
            //{
            //    throw;
            //}
        }
        /// <summary>
        /// Gets the corner points.
        /// </summary>
        /// <param name="upperLeft">The upper left.</param>
        /// <param name="lowerRight">The lower right.</param>
        /// <param name="elevation">The elevation.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public ICoordinate[] GetCornerPoints(GridCell upperLeft, GridCell lowerRight, double[] elevation)
        {
            try
            {
                double rX = 0.0;
                double rY = 0.0;
                double x;
                double y;

                if (upperLeft == null)
                    throw new NullReferenceException();
                if (lowerRight == null)
                    throw new NullReferenceException();
                if (upperLeft.Row < 1 || upperLeft.Row > RowCount || upperLeft.Column < 1 || upperLeft.Column > ColumnCount)
                    throw new ArgumentException("Invalid cell location.");
                if (lowerRight.Row < 1 || lowerRight.Row > RowCount || lowerRight.Column < 1 || lowerRight.Column > ColumnCount)
                    throw new ArgumentException("Invalid cell location.");

                List<ICoordinate> list = new List<ICoordinate>();

                // point 1
                x = Origin.X;
                if (upperLeft.Column > 1)
                { x += ColumnSpacing.GetEdgeOffset(upperLeft.Column - 1); }
                y = Origin.Y + (TotalRowHeight - RowSpacing.GetEdgeOffset(lowerRight.Row));
                if ((Angle == 0))
                {
                    rX = x;
                    rY = y;
                }
                else
                {
                    MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
                }
                list.Add((ICoordinate)(new Coordinate(rX, rY, elevation[0])));

                // point 2
                if (upperLeft.Row == 1)
                { y = Origin.Y + TotalRowHeight; }
                else
                { y = Origin.Y + (TotalRowHeight - RowSpacing.GetEdgeOffset(upperLeft.Row - 1)); }
                if ((Angle == 0))
                {
                    rX = x;
                    rY = y;
                }
                else
                {
                    MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
                }
                list.Add((ICoordinate)(new Coordinate(rX, rY, elevation[1])));

                // point 3
                x = Origin.X + ColumnSpacing.GetEdgeOffset(lowerRight.Column);
                if ((Angle == 0))
                {
                    rX = x;
                    rY = y;
                }
                else
                {
                    MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
                }
                list.Add((ICoordinate)(new Coordinate(rX, rY, elevation[2])));

                // point 4
                y = Origin.Y + (TotalRowHeight - RowSpacing.GetEdgeOffset(lowerRight.Row));
                if ((Angle == 0))
                {
                    rX = x;
                    rY = y;
                }
                else
                {
                    MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
                }
                list.Add((ICoordinate)(new Coordinate(rX, rY, elevation[3])));

                ICoordinate[] ptList = list.ToArray();
                return ptList;
            }
            catch (Exception ex)
            {
                throw;
            }

        }
        /// <summary>
        /// Gets the outline.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public IMultiLineString GetOutline()
        {
            return GetOutline(new GridCell(1, 1), new GridCell(this.RowCount, this.ColumnCount));
        }
        /// <summary>
        /// Gets the outline.
        /// </summary>
        /// <param name="cell">The cell.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public IMultiLineString GetOutline(GridCell cell)
        {
            return GetOutline(cell, new GridCell(cell.Row, cell.Column));
        }
        /// <summary>
        /// Gets the outline.
        /// </summary>
        /// <param name="upperLeft">The upper left.</param>
        /// <param name="lowerRight">The lower right.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public IMultiLineString GetOutline(GridCell upperLeft, GridCell lowerRight)
        {
            return GetOutline(upperLeft, lowerRight, 0.0);
        }
        /// <summary>
        /// Gets the outline.
        /// </summary>
        /// <param name="upperLeft">The upper left.</param>
        /// <param name="lowerRight">The lower right.</param>
        /// <param name="elevation">The elevation.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public IMultiLineString GetOutline(GridCell upperLeft, GridCell lowerRight, double elevation)
        {
            ICoordinate[] corners = this.GetCornerPoints(upperLeft, lowerRight);
            ICoordinate[] points = new ICoordinate[5];

            for (int i = 0; i < 4; i++)
            {
                corners[i].Z = elevation;
                points[i] = corners[i]; 
            }
            points[4] = corners[0];

            ILineString[] lines = new LineString[1];
            lines[0] = new LineString(points);
            IMultiLineString outline = new MultiLineString(lines);

            return outline;

        }
        /// <summary>
        /// Gets the outline.
        /// </summary>
        /// <param name="upperLeft">The upper left.</param>
        /// <param name="lowerRight">The lower right.</param>
        /// <param name="top">The top.</param>
        /// <param name="bottom">The bottom.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public IMultiLineString GetOutline(GridCell upperLeft, GridCell lowerRight, Array2d<float> top, Array2d<float> bottom)
        {
            ICoordinate[] cornersBottom = null;
            ICoordinate[] pointsBottom = null;
            ICoordinate[] cornersTop = null;
            ICoordinate[] pointsTop = null;
            ICoordinate[] pointsVerticalEdge0 = null;
            ICoordinate[] pointsVerticalEdge1 = null;
            ICoordinate[] pointsVerticalEdge2 = null;
            ICoordinate[] pointsVerticalEdge3 = null;

            GridCell lowerLeft = new GridCell(lowerRight.Row, upperLeft.Column);
            GridCell upperRight = new GridCell(upperLeft.Row, lowerRight.Column);
            double[] elevationBot = new double[4] { 0.0, 0.0, 0.0, 0.0 };
            double[] elevationTop = new double[4] { 0.0, 0.0, 0.0, 0.0 };

            if (bottom == null && top == null)
            {
                cornersTop = this.GetCornerPoints(upperLeft, lowerRight, elevationTop);
                pointsTop = new ICoordinate[5];
                for (int i = 0; i < 4; i++)
                {
                    pointsTop[i] = cornersTop[i];
                }
                pointsTop[4] = cornersTop[0];
            }
            else
            {
                if (bottom != null)
                {
                    elevationBot[0] = bottom[lowerLeft.Row, lowerLeft.Column];
                    elevationBot[1] = bottom[upperLeft.Row, upperLeft.Column];
                    elevationBot[2] = bottom[upperRight.Row, upperRight.Column];
                    elevationBot[3] = bottom[lowerRight.Row, lowerRight.Column];
                    cornersBottom = this.GetCornerPoints(upperLeft, lowerRight, elevationBot);
                    pointsBottom = new ICoordinate[5];
                    for (int i = 0; i < 4; i++)
                    {
                        pointsBottom[i] = cornersBottom[i];
                    }
                    pointsBottom[4] = cornersBottom[0];
                }

                if (top != null)
                {
                    elevationTop[0] = top[lowerLeft.Row, lowerLeft.Column];
                    elevationTop[1] = top[upperLeft.Row, upperLeft.Column];
                    elevationTop[2] = top[upperRight.Row, upperRight.Column];
                    elevationTop[3] = top[lowerRight.Row, lowerRight.Column];
                    cornersTop = this.GetCornerPoints(upperLeft, lowerRight, elevationTop);
                    pointsTop = new ICoordinate[5];
                    for (int i = 0; i < 4; i++)
                    {
                        pointsTop[i] = cornersTop[i];
                    }
                    pointsTop[4] = cornersTop[0];
                }

                if (pointsBottom != null && pointsTop != null)
                {
                    pointsVerticalEdge0 = new ICoordinate[2];
                    pointsVerticalEdge1 = new ICoordinate[2];
                    pointsVerticalEdge2 = new ICoordinate[2];
                    pointsVerticalEdge3 = new ICoordinate[2];
                    pointsVerticalEdge0[0] = new Coordinate(pointsBottom[0]);
                    pointsVerticalEdge0[1] = new Coordinate(pointsTop[0]);
                    pointsVerticalEdge1[0] = new Coordinate(pointsBottom[1]);
                    pointsVerticalEdge1[1] = new Coordinate(pointsTop[1]);
                    pointsVerticalEdge2[0] = new Coordinate(pointsBottom[2]);
                    pointsVerticalEdge2[1] = new Coordinate(pointsTop[2]);
                    pointsVerticalEdge3[0] = new Coordinate(pointsBottom[3]);
                    pointsVerticalEdge3[1] = new Coordinate(pointsTop[3]);
                }

            }

            ILineString[] lines;
            if (pointsBottom != null && pointsTop != null)
            {
                lines = new ILineString[6];
                lines[0] = new LineString(pointsBottom);
                lines[1] = new LineString(pointsTop);
                lines[2] = new LineString(pointsVerticalEdge0);
                lines[3] = new LineString(pointsVerticalEdge1);
                lines[4] = new LineString(pointsVerticalEdge2);
                lines[5] = new LineString(pointsVerticalEdge3);

            }
            else
            {
                lines = new ILineString[1];
                if (pointsBottom != null)
                {
                    lines[0] = new LineString(pointsBottom);
                }
                else if (pointsTop != null)
                {
                    lines[0] = new LineString(pointsTop);
                }
            }

            IMultiLineString outline = new MultiLineString(lines);
            return outline;

        }
        /// <summary>
        /// Gets the polygon.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public IPolygon GetPolygon()
        {
            return GetPolygon(new GridCell(1, 1), new GridCell(this.RowCount, this.ColumnCount));
        }
        /// <summary>
        /// Gets the polygon.
        /// </summary>
        /// <param name="cell">The cell.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public IPolygon GetPolygon(GridCell cell)
        {
            return GetPolygon(cell, new GridCell(cell.Row, cell.Column));
        }
        /// <summary>
        /// Gets the polygon.
        /// </summary>
        /// <param name="upperLeft">The upper left.</param>
        /// <param name="lowerRight">The lower right.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public IPolygon GetPolygon(GridCell upperLeft, GridCell lowerRight)
        {
            ICoordinate[] corners = this.GetCornerPoints(upperLeft, lowerRight);
            ICoordinate[] points = new ICoordinate[5];

            for (int i = 0; i < 4; i++)
            { points[i] = corners[i]; }
            points[4] = corners[0];

            IPolygon polygon = new Polygon(new LinearRing(points));
            return polygon;
        }

        /// <summary>
        /// Gets an array of lines representing the interior gridlines.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public IMultiLineString[] GetGridlines()
        {
            IMultiLineString line = null;
            List<IMultiLineString> lines = new List<IMultiLineString>();
            
            for (int row = 1; row <= this.RowCount; row++)
            {
                line = this.GetRowGridline(row);
                lines.Add(line);
            }

            for (int column = 1; column <= this.ColumnCount; column++)
            {
                line = this.GetColumnGridline(column);
                lines.Add(line);
            }

            return lines.ToArray();

        }


        /// <summary>
        /// Tries the get corner points.
        /// </summary>
        /// <param name="cell">The cell.</param>
        /// <param name="cornerPoints">The corner points.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public bool TryGetCornerPoints(GridCell cell, List<ICoordinate> cornerPoints)
        {
            try
            {
                ICoordinate pt = null;
                double rX = 0;
                double rY = 0;
                double x;
                double y;

                if (cornerPoints == null)
                    throw new NullReferenceException("corners");
                cornerPoints.Clear();

                if (cell == null)
                    throw new NullReferenceException();
                if (cell.Row < 1 || cell.Row > RowCount || cell.Column < 1 || cell.Column > ColumnCount)
                    throw new ArgumentException("Invalid cell location.");

                // point 1
                x = Origin.X;
                if (cell.Column > 1)
                { x += ColumnSpacing.GetEdgeOffset(cell.Column - 1); }
                y = Origin.Y + (TotalRowHeight - RowSpacing.GetEdgeOffset(cell.Row));
                if ((Angle == 0))
                {
                    rX = x;
                    rY = y;
                }
                else
                {
                    MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
                }
                pt = (ICoordinate)(new Coordinate(rX, rY));
                cornerPoints.Add(pt);

                // point 2
                if (cell.Row == 1)
                { y = Origin.Y + TotalRowHeight; }
                else
                { y = Origin.Y + (TotalRowHeight - RowSpacing.GetEdgeOffset(cell.Row - 1)); }
                if ((Angle == 0))
                {
                    rX = x;
                    rY = y;
                }
                else
                {
                    MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
                }
                pt = (ICoordinate)(new Coordinate(rX, rY));
                cornerPoints.Add(pt);

                // point 3
                x = Origin.X + ColumnSpacing.GetEdgeOffset(cell.Column);
                if ((Angle == 0))
                {
                    rX = x;
                    rY = y;
                }
                else
                {
                    MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
                }
                pt = (ICoordinate)(new Coordinate(rX, rY));
                cornerPoints.Add(pt);

                // point 4
                y = Origin.Y + (TotalRowHeight - RowSpacing.GetEdgeOffset(cell.Row));
                if ((Angle == 0))
                {
                    rX = x;
                    rY = y;
                }
                else
                {
                    MathUtility.FastForwardRotate(_AngleCos, _AngleSin, Origin.X, Origin.Y, x, y, ref rX, ref rY);
                }
                pt = (ICoordinate)(new Coordinate(rX, rY));
                cornerPoints.Add(pt);

                return true;

            }
            catch (Exception)
            {
                return false;
            }
        }

        /// <summary>
        /// Gets the grid lines.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public IMultiLineString[] GetGridLines()
        {
            return GetGridLines(new GridCell(1, 1), new GridCell(RowCount, ColumnCount), false);
        }
        /// <summary>
        /// Gets the grid lines.
        /// </summary>
        /// <param name="upperLeft">The upper left.</param>
        /// <param name="lowerRight">The lower right.</param>
        /// <param name="RegionIsHole">if set to <c>true</c> [region is hole].</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public IMultiLineString[] GetGridLines(GridCell upperLeft, GridCell lowerRight,bool RegionIsHole)
        {
            try
            {
                if (upperLeft == null)
                    throw new NullReferenceException();
                if (lowerRight == null)
                    throw new NullReferenceException();
                if (upperLeft.Row < 1 || upperLeft.Row > RowCount || upperLeft.Column < 1 || upperLeft.Column > ColumnCount)
                    throw new ArgumentException("Invalid cell location.");
                if (lowerRight.Row < 1 || lowerRight.Row > RowCount || lowerRight.Column < 1 || lowerRight.Column > ColumnCount)
                    throw new ArgumentException("Invalid cell location.");

                List<IMultiLineString> list = new List<IMultiLineString>();
                IMultiLineString mls = null;
                for (int row = upperLeft.Row-1; row <= lowerRight.Row; row++)
                {
                    mls = GetRowGridline(row);
                    if (mls != null) list.Add(mls);
                }

                for (int column = upperLeft.Column-1; column <= lowerRight.Column; column++)
                {
                    mls = GetColumnGridline(column);
                    if (mls != null) list.Add(mls);
                }

                IMultiLineString[] mlsArray = list.ToArray();
                return mlsArray;

            }
            catch (Exception ex)
            {
                throw;
            }            

        }

        #endregion

        #region IDataObject Members

        /// <summary>
        /// 
        /// </summary>
        private string _PumaType = "";
        /// <summary>
        /// Gets the fully qualified type name of this object.
        /// </summary>
        /// <remarks></remarks>
        public string PumaType
        {
            get
            {
                if (String.IsNullOrEmpty(_PumaType))
                {
                    _DefaultName = "CellCenteredArealGrid";
                    _PumaType = _DefaultName;
                }

                return _PumaType;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        private string _DefaultName = "";
        /// <summary>
        /// Gets the default name that will be used for the root XML element of this
        /// class.
        /// </summary>
        /// <remarks></remarks>
        public string DefaultName
        {
            get
            {
                if (String.IsNullOrEmpty(_DefaultName))
                {
                    _DefaultName = "CellCenteredArealGrid";
                    _PumaType = _DefaultName;
                }

                return _DefaultName;

            }
        }

        /// <summary>
        /// Gets the Puma version of the XML data format for this data object.
        /// </summary>
        /// <remarks></remarks>
        public int Version
        { get { return 1; } }

        /// <summary>
        /// Returns True if the DataObject is properly initialized.
        /// </summary>
        private bool m_IsValid = false;
        /// <summary>
        /// Gets a value indicating whether this instance is valid.
        /// </summary>
        /// <remarks></remarks>
        public bool IsValid
        {
            get { return m_IsValid; }
            private set { m_IsValid = value; }
        }


        #endregion 

        #region ISerializeXml Members
        /// <summary>
        /// Loads from XML.
        /// </summary>
        /// <param name="xmlString">The XML string.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public bool LoadFromXml(string xmlString)
        {
            try
            {
                double result = 0;
                double[] spacing = null;

                Angle = 0;
                RowSpacing.SetSize(0, 0);
                ColumnSpacing.SetSize(0, 0);

                System.Xml.XmlDocument doc = new System.Xml.XmlDocument();
                System.Xml.XmlNode node;

                doc.LoadXml(xmlString);
                System.Xml.XmlNode root = doc.DocumentElement;

                if (root == null)
                { throw new Exception("Error loading Puma object from XML."); }

                if (!DataObjectUtility.IsValidPumaType(root, PumaType))
                { throw new Exception("Error loading Puma object from XML. Incorrect PumaType."); }

                // Declare a list object to hold the grid spacing data
                IndexRangeValueList list = null;

                // Process rows
                node = root.SelectSingleNode("RowCount");
                int rowCount = int.Parse(node.InnerText);
                node = root.SelectSingleNode("RowSpacing");
                if (double.TryParse(node.InnerText,out result))
                { RowSpacing.SetSize(rowCount,result); }
                else
                {
                    spacing = NumberArrayIO<double>.StringToArray(node.InnerText);
                    if (spacing.Length != rowCount)
                    { throw (new Exception("Error computing row spacing from XML.")); }
                    RowSpacing.SetSize(spacing);
                }

                // Process columns
                node = root.SelectSingleNode("ColumnCount");
                int columnCount = int.Parse(node.InnerText);
                node = root.SelectSingleNode("ColumnSpacing");
                if (double.TryParse(node.InnerText,out result))
                { ColumnSpacing.SetSize(columnCount,result); }
                else
                {
                    spacing = NumberArrayIO<double>.StringToArray(node.InnerText);
                    if (spacing.Length != columnCount)
                    { throw (new Exception("Error computing column spacing from XML.")); }
                    ColumnSpacing.SetSize(spacing);
                }

                // Process Angle
                node = root.SelectSingleNode("Angle");
                Angle = double.Parse(node.InnerText);

                // Process Origin
                node = root.SelectSingleNode("Origin");
                // ToDo -- fix this to work with GeoAPI and NTS geometry libs
                //if (!origin.LoadFromXml(node.OuterXml))
                //{ throw new Exception("Error loading origin point for ArealGrid object."); }
                //double rx;
                //double ry;
                //Origin = new Point(rx, ry);

                // Process Projection
                Projection = "";
                node = root.SelectSingleNode("Projection");
                if (node != null) Projection = node.InnerText;

                IsValid = true;
                return true;

            }
            catch (Exception)
            {
                IsValid = false;
                return false;
            }
        }
        /// <summary>
        /// Saves as XML.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public string SaveAsXml()
        {
            return SaveAsXml(DefaultName);
        }
        /// <summary>
        /// Saves as XML.
        /// </summary>
        /// <param name="elementName">Name of the element.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public string SaveAsXml(string elementName)
        {
            try
            {
                double[] spacing = null;
                string sXml = null;
                System.Xml.XmlNode node = null;
                System.Xml.XmlDocumentFragment docFrag = null;
                System.Xml.XmlDocument doc = DataObjectUtility.XmlWrapperDoc(this, elementName);
                System.Xml.XmlNode root = doc.DocumentElement;

                // Append puma type
                DataObjectUtility.AppendPumaTypeAttribute(this, root);

                // Append the row count
                node = root.AppendChild(doc.CreateElement("RowCount"));
                node.InnerText = RowCount.ToString();

                // Append the column count
                node = root.AppendChild(doc.CreateElement("ColumnCount"));
                node.InnerText = ColumnCount.ToString();

                // Write the origin point as xml and append it to the root element
                // ToDo -- fix this to work with GeoAPI and NTS geometry libs
                //sXml = Origin.SaveAsXml("Origin");
                //docFrag = doc.CreateDocumentFragment();
                //docFrag.InnerXml = sXml;
                //node = root.AppendChild(docFrag);

                // Append the rotation angle.
                node = root.AppendChild(doc.CreateElement("Angle"));
                node.InnerText = Angle.ToString();

                // Append row spacing
                root.AppendChild(doc.CreateElement("RowSpacing"));
                if (RowSpacing.IsConstantSpacing)
                { root.LastChild.InnerText = RowSpacing[1].ToString(); }
                else
                {
                    spacing = new double[RowCount];
                    for (int i = 0; i < RowCount; i++)
                    { spacing[i] = RowSpacing[i + 1]; }
                    root.LastChild.InnerText = NumberArrayIO<double>.ArrayToString(spacing, true);
                }

                // Append column spacing
                root.AppendChild(doc.CreateElement("ColumnSpacing"));
                if (ColumnSpacing.IsConstantSpacing)
                { root.LastChild.InnerText = ColumnSpacing[1].ToString(); }
                else
                {
                    spacing = new double[ColumnCount];
                    for (int i = 0; i < ColumnCount; i++)
                    { spacing[i] = ColumnSpacing[i + 1]; }
                    root.LastChild.InnerText = NumberArrayIO<double>.ArrayToString(spacing, true);
                }

                // Append projection
                root.AppendChild(doc.CreateElement("Projection"));
                root.LastChild.InnerText = Projection;

                // Return the result as an xml string.
                return root.OuterXml;


            }
            catch (Exception)
            {
                throw new Exception("Error saving Puma object as XML.");
            }
        }
        #endregion


        #endregion

        #region Private Properties
        /// <summary>
        /// Gets an instance of GridSpacing class for grid rows to be used internally.
        /// </summary>
        /// <remarks></remarks>
        private GridSpacing RowSpacing
        { get { return _Rows; } }
        /// <summary>
        /// Gets an instance of GridSpacing class for grid columns to be used internally.
        /// </summary>
        /// <remarks></remarks>
        private GridSpacing ColumnSpacing
        { get { return _Columns; } }

        #endregion

        #region Private Methods
        /// <summary>
        /// Normalizes the angle.
        /// </summary>
        /// <param name="AngleValue">The angle value.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private double NormalizeAngle(double AngleValue)
        {
            double r = 0;
            double r360 = 360;
            double r180 = 180;

            try
            {
                if ((AngleValue >= r360) && (AngleValue <= -r360))
                { r = AngleValue % r360; }
                else
                { r = AngleValue; }

                if (r > r180)
                { r = r - r360; }
                else if (r < -r180)
                { r = r + r360; }

                return r;

            }
            catch
            { return 0; }

        }
        /// <summary>
        /// Computes the cell count.
        /// </summary>
        /// <param name="SingleCellSize">Size of the single cell.</param>
        /// <param name="TotalLength">The total length.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private int ComputeCellCount(double SingleCellSize, double TotalLength)
        {
            int n = 0;

            try
            {
                n = System.Convert.ToInt32(Math.Round(TotalLength / SingleCellSize, 0));
                if (n < 1) n = 1;
                return n;
            }
            catch (Exception ex)
            {
                return 0;
            }

        }
        /// <summary>
        /// Initializes the square cell grid.
        /// </summary>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <param name="cellSize">Size of the cell.</param>
        /// <param name="origin">The origin.</param>
        /// <param name="rotationAngle">The rotation angle.</param>
        /// <remarks></remarks>
        private void InitializeSquareCellGrid(int rows, int columns, double cellSize, ICoordinate origin, double rotationAngle)
        {
            if (origin == null)
            { InitializeSquareCellGrid(rows, columns, cellSize, 0, 0, rotationAngle); }
            else
            {
                InitializeSquareCellGrid(rows, columns, cellSize, origin.X, origin.Y, rotationAngle);
            }
        }
        /// <summary>
        /// Initializes the square cell grid.
        /// </summary>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <param name="cellSize">Size of the cell.</param>
        /// <param name="originX">The origin X.</param>
        /// <param name="originY">The origin Y.</param>
        /// <param name="rotationAngle">The rotation angle.</param>
        /// <remarks></remarks>
        private void InitializeSquareCellGrid(int rows, int columns, double cellSize, double originX,double originY, double rotationAngle)
        {
            try
            {
                Angle = rotationAngle;
                if (_Origin == null)
                { _Origin = (ICoordinate)(new Coordinate(originX, originY)); }
                else
                {
                    _Origin.X = originX;
                    _Origin.Y = originY;
                }

                if (rows < 1)
                { _Rows = new GridSpacing(); }
                else
                { _Rows = new GridSpacing(rows, cellSize); }

                if (columns < 1)
                { _Columns = new GridSpacing(); }
                else
                { _Columns = new GridSpacing(columns, cellSize); }

                IsValid = true;

            }
            catch (Exception)
            {
                IsValid = false;
                throw new Exception("Initialization error during creation of ArealGrid object.");
            }

        }

        #endregion

        #endregion

        #region Internal Class: GridSpacing
        /// <summary>
        /// GridSpacing is a private internal class that has methods and logic
        /// for managing things related to row and column dimensions and coordinates.
        /// </summary>
        /// <remarks></remarks>
        private class GridSpacing
        {
            #region Fields
            /// <summary>
            /// 
            /// </summary>
            private System.Collections.Generic.List<double> _Spacing;
            /// <summary>
            /// 
            /// </summary>
            private System.Collections.Generic.List<double> _EdgeOffsets;
            /// <summary>
            /// 
            /// </summary>
            private double _DefaultSpacing = 0;
            #endregion

            #region Events and event methods
            /// <summary>
            /// Occurs when [default spacing changed].
            /// </summary>
            /// <remarks></remarks>
            public event System.EventHandler DefaultSpacingChanged;
            /// <summary>
            /// Called when [default spacing changed].
            /// </summary>
            /// <remarks></remarks>
            private void OnDefaultSpacingChanged()
            {
                if (DefaultSpacingChanged != null)
                {
                    DefaultSpacingChanged(this, new System.EventArgs());
                }
            }
            #endregion

            #region Constructors
            /// <summary>
            /// Initializes a new instance of the <see cref="T:System.Object"/> class.
            /// </summary>
            /// <remarks></remarks>
            public GridSpacing()
            {
                _Spacing = new List<double>();
                _EdgeOffsets = new List<double>();
                ResetArrays();
            }
            /// <summary>
            /// Initializes a new instance of the <see cref="GridSpacing"/> class.
            /// </summary>
            /// <param name="size">The size.</param>
            /// <param name="spacing">The spacing.</param>
            /// <remarks></remarks>
            public GridSpacing(int size, double spacing)
            {
                _Spacing = new List<double>();
                _EdgeOffsets = new List<double>();
                SetSize(size,spacing);
            }
            /// <summary>
            /// Initializes a new instance of the <see cref="GridSpacing"/> class.
            /// </summary>
            /// <param name="spacing">The spacing.</param>
            /// <remarks></remarks>
            public GridSpacing(double[] spacing)
            {
                _Spacing = new List<double>();
                _EdgeOffsets = new List<double>();
                SetSize(spacing);
            }
            #endregion

            #region Public Properties

            /// <summary>
            /// Gets or sets the <see cref="System.Double"/> at the specified index.
            /// </summary>
            /// <remarks></remarks>
            public double this[int index]
            {
                get
                {
                    if (index < 1 || index > Count) throw new Exception("Index is out of range.");
                    return _Spacing[index]; 
                }
                set
                {
                    if (index < 1 || index > Count) throw new Exception("Index is out of range.");
                    if (_Spacing[index] == value) return;
                    _Spacing[index] = value;
                    BuildEdgeOffsets();
                }

            }

            /// <summary>
            /// Gets the count.
            /// </summary>
            /// <remarks></remarks>
            public int Count
            { get 
              { 
                return _Spacing.Count-1; 
              } 
            }

            /// <summary>
            /// Gets the total length.
            /// </summary>
            /// <remarks></remarks>
            public double TotalLength
            {
                get
                { return _EdgeOffsets[Count]; }
            }
            #endregion

            #region Public Methods
            /// <summary>
            /// Gets a value indicating whether this instance is constant spacing.
            /// </summary>
            /// <remarks></remarks>
            public bool IsConstantSpacing
            {
                get
                {
                    try
                    {
                        double first = _Spacing[1];
                        for (int i = 2; i <= Count; i++)
                        {
                            if (first != _Spacing[i]) return false;
                        }
                        return true;
                    }
                    catch (Exception ex)
                    {
                        return false;
                    }
                }
            }
            /// <summary>
            /// Sets the size.
            /// </summary>
            /// <param name="size">The size.</param>
            /// <param name="defaultSpacing">The default spacing.</param>
            /// <param name="regionList">The region list.</param>
            /// <remarks></remarks>
            public void SetSize(int size, double defaultSpacing, IndexRangeValueList regionList)
            {
                try
                {
                    if (size < 1) return;
                    SetSize(size, defaultSpacing);
                    if (regionList == null) return;
                    if (regionList.Count == 0) return;
                    foreach (IndexRangeValue item in regionList)
                    {
                        for (int i = item.FromIndex; i <= item.ToIndex; i++)
                        {
                            if (i > 0 && i <= Count) _Spacing[i] = item.DataValue;
                        }
                    }
                    BuildEdgeOffsets();
                }
                catch (Exception)
                {
                    throw new Exception("Error setting grid size spacing.");
                }
            }
            /// <summary>
            /// Sets the size.
            /// </summary>
            /// <param name="size">The size.</param>
            /// <param name="spacing">The spacing.</param>
            /// <remarks></remarks>
            public void SetSize(int size, double spacing)
            {
                ResetArrays();
                for (int i = 0; i < size; i++)
                { AppendItem(spacing); }
            }
            /// <summary>
            /// Sets the size.
            /// </summary>
            /// <param name="spacing">The spacing.</param>
            /// <remarks></remarks>
            public void SetSize(double[] spacing)
            {
                if (spacing != null)
                {
                    int size = spacing.Length;
                    ResetArrays();

                    for (int i = 0; i < size; i++)
                    {
                        AppendItem(spacing[i]);
                    }
                }
            }
            /// <summary>
            /// Sets the size.
            /// </summary>
            /// <param name="spacing">The spacing.</param>
            /// <remarks></remarks>
            public void SetSize(double spacing)
            {
                SetSize(Count, spacing);
            }
            /// <summary>
            /// Sets the size.
            /// </summary>
            /// <param name="spacing">The spacing.</param>
            /// <param name="totalLength">The total length.</param>
            /// <remarks></remarks>
            public void SetSize(double spacing, double totalLength)
            {
                try
                {
                    int n = ComputeCellCount(spacing, totalLength);
                    SetSize(n, spacing);
                }
                catch (Exception)
                {
                    // return
                }
            }
            /// <summary>
            /// Adds the specified spacing.
            /// </summary>
            /// <param name="spacing">The spacing.</param>
            /// <remarks></remarks>
            public void Add(double spacing)
            { AppendItem(spacing); }
            /// <summary>
            /// Adds the specified spacing.
            /// </summary>
            /// <param name="spacing">The spacing.</param>
            /// <remarks></remarks>
            public void Add(double[] spacing)
            {
                try
                {
                    if (spacing == null) return;
                    for (int i = 0; i < spacing.Length; i++)
                    { AppendItem(spacing[i]); }
                }
                catch (Exception)
                {
                    // throw exception
                }

            }
            /// <summary>
            /// Inserts the specified index.
            /// </summary>
            /// <param name="index">The index.</param>
            /// <remarks></remarks>
            public void Insert(int index)
            {
                try
                { InsertItem(index, _DefaultSpacing); }
                catch (System.IndexOutOfRangeException)
                { throw new System.ArgumentOutOfRangeException("layer", "Specified layer is out of range."); }
            }
            /// <summary>
            /// Inserts the specified index.
            /// </summary>
            /// <param name="index">The index.</param>
            /// <param name="spacing">The spacing.</param>
            /// <remarks></remarks>
            public void Insert(int index, double spacing)
            {
                try
                { InsertItem(index, spacing); }
                catch (System.IndexOutOfRangeException)
                { throw new System.ArgumentOutOfRangeException("layer", "Specified layer is out of range."); }
            }
            /// <summary>
            /// Inserts the specified index.
            /// </summary>
            /// <param name="index">The index.</param>
            /// <param name="spacing">The spacing.</param>
            /// <remarks></remarks>
            public void Insert(int index, double[] spacing)
            {
                try
                {
                    if (spacing == null) return;
                    for (int i = 0; i < spacing.Length; i++)
                    { InsertItem(index + i, spacing[i]); }
                }
                catch (Exception)
                {
                    // throw exception
                }
            }
            /// <summary>
            /// Removes the last.
            /// </summary>
            /// <remarks></remarks>
            public void RemoveLast()
            {
                try
                { RemoveItem(_Spacing.Count); }
                catch (System.IndexOutOfRangeException)
                { throw new System.InvalidOperationException("Error attempting to remove an array element."); }
            }
            /// <summary>
            /// Removes the specified index.
            /// </summary>
            /// <param name="index">The index.</param>
            /// <remarks></remarks>
            public void Remove(int index)
            {
                try
                { RemoveItem(index); }
                catch (System.IndexOutOfRangeException)
                { throw new System.ArgumentOutOfRangeException("layer", "Specified layer is out of range."); }
            }
            /// <summary>
            /// Gets the center offset.
            /// </summary>
            /// <param name="index">The index.</param>
            /// <returns></returns>
            /// <remarks></remarks>
            public double GetCenterOffset(int index)
            {
                try
                {
                    return _EdgeOffsets[index] - _Spacing[index] / 2; 
                }
                catch (System.IndexOutOfRangeException)
                { throw new System.ArgumentOutOfRangeException("layer", "Specified layer is out of range."); }
            }
            /// <summary>
            /// Gets the edge offset.
            /// </summary>
            /// <param name="index">The index.</param>
            /// <returns></returns>
            /// <remarks></remarks>
            public double GetEdgeOffset(int index)
            {
                try
                { return _EdgeOffsets[index]; }
                catch (System.IndexOutOfRangeException)
                { throw new System.ArgumentOutOfRangeException("layer", "Specified layer is out of range."); }
            }
            /// <summary>
            /// Finds the index of the cell.
            /// </summary>
            /// <param name="coordValue">The coord value.</param>
            /// <returns></returns>
            /// <remarks></remarks>
            public int FindCellIndex(double coordValue)
            {
                try
                {
                    if (coordValue < 0) return 0;
                    if (coordValue > TotalLength) return 0;

                    for (int i = 1; i <= Count; i++)
                    { if (coordValue <= _EdgeOffsets[i]) return i; }

                    return 0;

                }
                catch (Exception)
                {
                    return 0;
                }
            }
            /// <summary>
            /// Gets the compact spacing.
            /// </summary>
            /// <returns></returns>
            /// <remarks></remarks>
            public IndexRangeValueList GetCompactSpacing()
            {
                try
                {
                    IndexRangeValue gridCellRegion = null;
                    IndexRangeValueList list = new IndexRangeValueList();
                    
                    for (int i = 1; i <= Count; i++)
                    {
                        if (i == 1)
                        {
                            gridCellRegion = new IndexRangeValue(i, i, _Spacing[i]);
                        }

                        if (_Spacing[i] != gridCellRegion.DataValue)
                        {
                            gridCellRegion.ToIndex = i - 1;
                            list.Add(gridCellRegion);
                            gridCellRegion = new IndexRangeValue(i, i, _Spacing[i]);
                        }

                        if (i == Count)
                        {
                            gridCellRegion.ToIndex = Count;
                            list.Add(gridCellRegion);
                        }
                    }

                    return list;

                }
                catch (Exception)
                {
                    return null;
                }
            }
            #endregion

            #region Private Methods
            /// <summary>
            /// Computes the cell count.
            /// </summary>
            /// <param name="cellSize">Size of the cell.</param>
            /// <param name="totalLength">The total length.</param>
            /// <returns></returns>
            /// <remarks></remarks>
            private int ComputeCellCount(double cellSize, double totalLength)
            {
                int n = 0;

                try
                {
                    n = System.Convert.ToInt32(Math.Round(totalLength / cellSize, 0));
                    if (n < 1) n = 1;
                    return n;
                }
                catch (Exception ex)
                {
                    return 0;
                }

            }
            /// <summary>
            /// Fills the spacing array.
            /// </summary>
            /// <param name="value">The value.</param>
            /// <remarks></remarks>
            private void FillSpacingArray(double value)
            {
                _Spacing[0] = 0;
                for (int i = 1; i <= Count; i++)
                {
                    _Spacing[i] = value;
                }
                BuildEdgeOffsets();

            }
            /// <summary>
            /// Builds the edge offsets.
            /// </summary>
            /// <remarks></remarks>
            private void BuildEdgeOffsets()
            {
                double lastOffset = 0;
                if (_Spacing.Count == _EdgeOffsets.Count)
                {
                    for (int i = 1; i <= Count; i++)
                    {
                        _EdgeOffsets[i] = _EdgeOffsets[i-1] + _Spacing[i];
                    }
                }

            }
            /// <summary>
            /// Appends the item.
            /// </summary>
            /// <param name="spacing">The spacing.</param>
            /// <remarks></remarks>
            private void AppendItem(double spacing)
            {
                _Spacing.Add(spacing);
                _EdgeOffsets.Add(_EdgeOffsets[Count-1] + spacing);
            }
            /// <summary>
            /// Inserts the item.
            /// </summary>
            /// <param name="index">The index.</param>
            /// <param name="spacing">The spacing.</param>
            /// <remarks></remarks>
            private void InsertItem(int index, double spacing)
            {
                if (index < 1) return;
                if (index > Count) return;
                _Spacing.Insert(index, spacing);
                _EdgeOffsets.Insert(index, 0);
                BuildEdgeOffsets();
            }
            /// <summary>
            /// Removes the item.
            /// </summary>
            /// <param name="index">The index.</param>
            /// <remarks></remarks>
            private void RemoveItem(int index)
            {
                try
                {
                    bool rebuiltOffsets = true;
                    if (index == Count) rebuiltOffsets = false;
                    _Spacing.RemoveAt(index);
                    _EdgeOffsets.RemoveAt(index);
                    if (rebuiltOffsets) BuildEdgeOffsets();
                }
                catch { }
            }
            /// <summary>
            /// Resets the arrays.
            /// </summary>
            /// <remarks></remarks>
            private void ResetArrays()
            {
                _Spacing.Clear();
                _EdgeOffsets.Clear();
                _Spacing.Add(0);
                _EdgeOffsets.Add(0);
            }
            #endregion

        }
        #endregion

    }
}
