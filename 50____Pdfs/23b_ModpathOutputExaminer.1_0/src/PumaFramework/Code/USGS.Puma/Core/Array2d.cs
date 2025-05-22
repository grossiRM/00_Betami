using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Core;
using USGS.Puma.Utilities;
using USGS.Puma.IO;

namespace USGS.Puma.Core
{
    /// <summary>
    /// Generic two-dimensional number array. Number types Int32(int), Single(float),
    /// and Double are supported. Any other data type will result in a runtime
    /// exception. The arrays are implemented as 1D arrays that can be accessed as
    /// 2D arrays using the row and column indices. They also can be accessed directly
    /// with a single cellNumber index.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <remarks></remarks>
    public class Array2d<T> : NumberArray<T>, IDataObject
    {
        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="Array2d&lt;T&gt;"/> class.
        /// </summary>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <remarks></remarks>
        public Array2d(int rows, int columns)
        {
            CreateNewArray(rows, columns);
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="Array2d&lt;T&gt;"/> class.
        /// </summary>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <param name="initialValue">The initial value.</param>
        /// <remarks></remarks>
        public Array2d(int rows, int columns,T initialValue)
        {
            CreateNewArray(rows, columns, initialValue);
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="Array2d&lt;T&gt;"/> class.
        /// </summary>
        /// <param name="matrixData">The matrix data.</param>
        /// <remarks></remarks>
        public Array2d(Array2d<T> matrixData)
        {
            CreateNewArray(matrixData);
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="Array2d&lt;T&gt;"/> class.
        /// </summary>
        /// <param name="xmlString">The XML string.</param>
        /// <remarks></remarks>
        public Array2d(string xmlString)
        {
            bool result = LoadFromXml(xmlString);
        }
        #endregion

        #region Public Members

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
                    T obj = default(T);
                    _PumaType= "Array2d" + obj.GetType().Name;
                    _DefaultName = _PumaType;
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
                    T obj = default(T);
                    _PumaType= "Array2d" + obj.GetType().Name;
                    _DefaultName = PumaType;
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
                System.Xml.XmlDocument doc = new System.Xml.XmlDocument();
                System.Xml.XmlNode node;
                System.Xml.XmlNode rowNode;
                System.Xml.XmlNodeList rowNodes;
                System.Xml.XmlNode attrNode;

                doc.LoadXml(xmlString);
                System.Xml.XmlNode root = doc.DocumentElement;

                if (root == null)
                { throw new Exception("Error loading Puma object from XML."); }

                //if (!DataObjectUtility.IsValidPumaType(root, PumaType))
                //{ throw new Exception("Error loading Puma object from XML. Incorrect PumaType."); }

                attrNode = root.Attributes.GetNamedItem("RowCount");
                int rows = int.Parse(attrNode.InnerText);
                attrNode = root.Attributes.GetNamedItem("ColumnCount");
                int columns = int.Parse(attrNode.InnerText);

                rowNodes = root.SelectNodes("Row");
                bool isConstant = false;
                if (rowNodes == null)
                { isConstant = true; }
                else
                { if (rowNodes.Count == 0) isConstant = true; }

                if (isConstant)
                {
                    T val;
                    node = root.SelectSingleNode("ConstantValue");
                    if ( !this.Math.TryParse(node.InnerText, out val) )
                    { val = default(T); }
                    CreateNewArray(rows, columns, val);
                }
                else
                {
                    CreateNewArray(rows, columns);
                    if (rowNodes.Count != RowCount)
                    { throw (new Exception("Error loading array data from xml.")); }

                    T[] rowArray;
                    for (int i = 0; i < RowCount; i++)
                    {
                        rowNode = rowNodes[i];
                        rowArray = NumberArrayIO<T>.StringToArray(rowNode.InnerText);
                        for (int column = 1; column <= ColumnCount; column++)
                        { this[i+1, column] = rowArray[column - 1]; }
                    }
                }

                IsValid = true;
                return true;

            }
            catch (Exception ex)
            {
                IsValid = false;
                //System.Windows.Forms.MessageBox.Show(ex.Message);
                throw;
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
                System.Xml.XmlNode node = null;
                System.Xml.XmlDocument doc = DataObjectUtility.XmlWrapperDoc(this, elementName);
                System.Xml.XmlNode root = doc.DocumentElement;

                System.Xml.XmlNode attNode = null;
                attNode = root.Attributes.SetNamedItem(doc.CreateAttribute("RowCount"));
                attNode.InnerText = RowCount.ToString();
                attNode = root.Attributes.SetNamedItem(doc.CreateAttribute("ColumnCount"));
                attNode.InnerText = ColumnCount.ToString();
                DataObjectUtility.AppendPumaTypeAttribute(this, root);
               
                if (IsConstant)
                {
                    T val = this[1, 1];
                    root.AppendChild(doc.CreateElement("ConstantValue"));
                    root.LastChild.InnerText = val.ToString();
                }
                else
                {
                    for (int row = 1; row <= RowCount; row++)
                    {
                        root.AppendChild(doc.CreateElement("Row"));
                        root.LastChild.InnerText = NumberArrayIO<T>.ArrayToString(ToRowArray(row), false);
                        attNode = root.LastChild.Attributes.SetNamedItem(doc.CreateAttribute("i"));
                        attNode.InnerText = row.ToString();
                   }
                }

                return root.OuterXml;


            }
            catch (Exception ex)
            {

                throw;
            }
        }

        /// <summary>
        /// Writes the specified writer.
        /// </summary>
        /// <param name="writer">The writer.</param>
        /// <param name="elementName">Name of the element.</param>
        /// <param name="writeDimensions">if set to <c>true</c> [write dimensions].</param>
        /// <remarks></remarks>
        public void Write(System.Xml.XmlWriter writer, string elementName, bool writeDimensions)
        {
            // Write the start tag for the root element
            writer.WriteStartElement(elementName);

            // Write the row and column dimension attribute if necessary
            if (writeDimensions)
            {
                writer.WriteAttributeString("RowCount", this.RowCount.ToString());
                writer.WriteAttributeString("ColumnCount", this.ColumnCount.ToString());
            }

            if (this.IsConstant)
            {
                writer.WriteElementString("ConstantValue", this[1, 1].ToString());
            }
            else
            {
                // Write each row as a single element of comma delimited numbers
                for (int row = 1; row <= this.RowCount; row++)
                {
                    writer.WriteStartElement("Row");
                    writer.WriteAttributeString("i", row.ToString());
                    writer.WriteString(NumberArrayIO<T>.ArrayToString(this.ToRowArray(row), false));
                    writer.WriteEndElement();
                    //writer.WriteElementString("Row", NumberArrayIO<T>.ArrayToString(this.ToRowArray(row), false));
                }
            }

            // Close the root element
            writer.WriteEndElement();

        }

        #endregion

        /// <summary>
        /// Get or set an array element by row and column number indices.
        /// The rowNumber and columnNumber indices start at 1.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public T this[int rowNumber, int columnNumber]
        {
            get 
            { 
                return this.BaseValues[(rowNumber - 1) * _ColumnCount + columnNumber - 1]; 
            }
            set 
            { 
                this.BaseValues[(rowNumber - 1) * _ColumnCount + columnNumber - 1] = value; 
            }
        }

        /// <summary>
        /// 
        /// </summary>
        private int _RowCount;
        /// <summary>
        /// Gets the number of rows in the array.
        /// </summary>
        /// <remarks></remarks>
        public int RowCount
        {
            get { return _RowCount; }
        }

        /// <summary>
        /// 
        /// </summary>
        private int _ColumnCount;
        /// <summary>
        /// Gets the number of columns in the array.
        /// </summary>
        /// <remarks></remarks>
        public int ColumnCount
        {
            get { return _ColumnCount; }
        }

        /// <summary>
        /// Set the element values of this array to the corresponding elements
        /// of the input array.
        /// </summary>
        /// <param name="a">A.</param>
        /// <remarks></remarks>
        public void SetValues(Array2d<T> a)
        {
            if (a == null)
            { throw (new ArgumentNullException("a"));}
            if ( a.RowCount != RowCount || a.ColumnCount != ColumnCount )
            { throw (new ArgumentException("Dimensions of the input array do not match.")); }

            this.SetValuesFromNumberArray(a as NumberArray<T>);

            //for (int row = 1; row <= RowCount; row++)
            //{
            //    for (int column = 1; column <= ColumnCount; column++)
            //    { this[row, column] = a[row, column]; }
            //}
        }

        /// <summary>
        /// Gets a copy of this array.
        /// </summary>
        /// <returns>An new instance of the <see cref="Array2d&lt;T&gt;"/> class.</returns>
        /// <remarks>This method always returns an array of type T values. Methods   and   can be used
        /// to obtain single precision and double precision arrays for any instance of <see cref="Array2d&lt;T&gt;"/></remarks>
        public Array2d<T> GetCopy()
        {
            Array2d<T> a = new Array2d<T>(RowCount, ColumnCount);
            for (int row = 1; row <= RowCount; row++)
            {
                for (int column = 1; column <= ColumnCount; column++)
                { a[row, column] = this[row, column]; }
            }
            return a;
        }
        /// <summary>
        /// Gets the copy of the array as single precision.
        /// </summary>
        /// <returns>A single precision array.</returns>
        /// <remarks></remarks>
        public Array2d<Single> GetCopyAsSingle()
        {
            Array2d<float> a = new Array2d<float>(this.RowCount, this.ColumnCount);
            for (int i = 1; i <= this.ElementCount; i++)
            {
                a[i] = this.GetValueAsSingle(i);
            }
            return a;
        }
        /// <summary>
        /// Gets the copy as double precision.
        /// </summary>
        /// <returns>A double precision array.</returns>
        /// <remarks></remarks>
        public Array2d<double> GetCopyAsDouble()
        {
            Array2d<double> a = new Array2d<double>(this.RowCount, this.ColumnCount);
            for (int i = 1; i <= this.ElementCount; i++)
            {
                a[i] = this.GetValueAsDouble(i);
            }
            return a;
        }
        /// <summary>
        /// Toes the row array.
        /// </summary>
        /// <param name="row">The row.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public T[] ToRowArray(int row)
        {
            if (row < 1 || row > RowCount) return null;
            T[] rowArray = new T[ColumnCount];
            for (int column = 1; column <= ColumnCount; column++)
            { rowArray[column - 1] = this[row, column]; }
            return rowArray;
        }

        /// <summary>
        /// Toes the array.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public T[,] ToArray()
        {
            T[,] mat = new T[RowCount, ColumnCount];
            for (int row = 1; row <= RowCount; row++)
            {
                for (int column = 1; column <= ColumnCount; column++)
                { mat[row - 1, column - 1] = this[row, column]; }
            }
            return mat;
        }

        #region Math Members
        /// <summary>
        /// Adds elements of the input matrix to the corresponding elements
        /// of this matrix.
        /// </summary>
        /// <param name="m">The m.</param>
        /// <remarks></remarks>
        public void Add(Array2d<T> m)
        {
            if (m.RowCount != RowCount || m.ColumnCount != ColumnCount)
            { throw (new ArgumentException("Input matrix dimensions are incorrect.")); }

            this.AddNumberArray(m as NumberArray<T>);
            
            //for (int n = 1; n < this.ElementCount + 1; n++)
            //{
            //    this[n] = _gn.Add(this[n], m[n]);
            //}
        }

        /// <summary>
        /// Subtracts the specified m.
        /// </summary>
        /// <param name="m">The m.</param>
        /// <remarks></remarks>
        public void Subtract(Array2d<T> m)
        {
            if (m.RowCount != RowCount || m.ColumnCount != ColumnCount)
            { throw (new ArgumentException("Input matrix dimensions are incorrect.")); }

            this.SubtractNumberArray(m as NumberArray<T>);

            //for (int row = 1; row <= RowCount; row++)
            //{
            //    for (int column = 1; column <= ColumnCount; column++)
            //    { this[row, column] = _gn.Subtract(this[row, column], m[row, column]); }
            //}
        }

        /// <summary>
        /// Multiply each element of this matrix by the corresponding element
        /// of the input matrix.
        /// </summary>
        /// <param name="m">The m.</param>
        /// <remarks></remarks>
        public void Multiply(Array2d<T> m)
        {
            if (m.RowCount != RowCount || m.ColumnCount != ColumnCount)
            { throw (new ArgumentException("Input matrix dimensions are incorrect.")); }

            this.MultiplyNumberArray(m as NumberArray<T>);

            //for (int row = 1; row <= RowCount; row++)
            //{
            //    for (int column = 1; column <= ColumnCount; column++)
            //    { this[row, column] = _gn.Multiply(this[row, column], m[row, column]); }
            //}
        }

        /// <summary>
        /// Divide each element of this matrix by the corresponding element
        /// of the input matrix.
        /// </summary>
        /// <param name="m">The m.</param>
        /// <remarks></remarks>
        public void Divide(Array2d<T> m)
        {
            if (m.RowCount != RowCount || m.ColumnCount != ColumnCount)
            { throw (new ArgumentException("Input matrix dimensions are incorrect.")); }
            
            this.DivideNumberArray(m as NumberArray<T>);

            //for (int row = 1; row <= RowCount; row++)
            //{
            //    for (int column = 1; column <= ColumnCount; column++)
            //    { this[row, column] = _gn.Divide(this[row, column], m[row, column]); }
            //}
        }

        #endregion

        #endregion

        #region Private Members
        /// <summary>
        /// Creates the new array.
        /// </summary>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <remarks></remarks>
        private void CreateNewArray(int rows, int columns)
        {
            T val = default(T);
            CreateNewArray(rows, columns, val);
        }
        /// <summary>
        /// Creates the new array.
        /// </summary>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <param name="initialValue">The initial value.</param>
        /// <remarks></remarks>
        private void CreateNewArray(int rows, int columns, T initialValue)
        {
            _RowCount = rows;
            _ColumnCount = columns;
            int elementCount = rows * columns;
            this.ResizeArray(elementCount, initialValue);
        }
        /// <summary>
        /// Creates the new array.
        /// </summary>
        /// <param name="matrixData">The matrix data.</param>
        /// <remarks></remarks>
        private void CreateNewArray(Array2d<T> matrixData)
        {
            if (matrixData == null)
            { throw (new ArgumentNullException()); }
            _RowCount = matrixData.RowCount;
            _ColumnCount = matrixData.ColumnCount;
            this.ResizeArray(matrixData.BaseValues);
        }

        #endregion





    }
}
