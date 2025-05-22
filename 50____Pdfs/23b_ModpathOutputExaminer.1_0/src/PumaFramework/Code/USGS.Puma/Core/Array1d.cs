using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Core;
using USGS.Puma.Utilities;
using USGS.Puma.IO;

namespace USGS.Puma.Core
{
    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <remarks></remarks>
    public class Array1d<T> : NumberArray<T>, IDataObject
    {
        #region Constructors
        /// <summary>
        /// Initializes an instance of class Array1d(T)
        /// </summary>
        /// <param name="elements">The elements.</param>
        /// <remarks></remarks>
        public Array1d(int elements)
        {
            CreateNewArray(elements);
        }
        /// <summary>
        /// Initializes an instance of class Array1d(T)
        /// </summary>
        /// <param name="elements">The elements.</param>
        /// <param name="initialValue">The initial value.</param>
        /// <remarks></remarks>
        public Array1d(int elements, T initialValue)
        {
            CreateNewArray(elements, initialValue);
        }
        /// <summary>
        /// Initializes an instance of class Array1d(T)
        /// </summary>
        /// <param name="arrayData">The array data.</param>
        /// <remarks></remarks>
        public Array1d(Array1d<T> arrayData)
        {
            CreateNewArray(arrayData);
        }
        /// <summary>
        /// Initializes an instance of class Array1d(T)
        /// </summary>
        /// <param name="xmlString">The XML string.</param>
        /// <remarks></remarks>
        public Array1d(string xmlString)
        {
            bool result = LoadFromXml(xmlString);
        }
        #endregion

        #region Public Members
        ///// <summary>
        ///// Gets a copy of the array values as a 1-D array.
        ///// </summary>
        ///// <returns></returns>
        ///// <remarks></remarks>
        //public T[] GetValues()
        //{
        //    T[] values = new T[ElementCount];
        //    for (int i = 0; i < ElementCount; i++)
        //    { values[i] = _Values[i]; }
        //    return values;
        //}

        ///// <summary>
        ///// Sets the array values from a 1-D array.
        ///// </summary>
        ///// <param name="values">The values.</param>
        ///// <remarks></remarks>
        //public void SetValues(T[] values)
        //{
        //    if (values.Length != ElementCount)
        //    { throw (new ArgumentException("Array dimensions do not match.")); }

        //    for (int i = 0; i < ElementCount; i++)
        //    { _Values[i] = values[i]; }

        //}
        ///// <summary>
        ///// 
        ///// </summary>
        //private int _ElementCount;
        ///// <summary>
        ///// Gets the element count.
        ///// </summary>
        ///// <remarks></remarks>
        //public int ElementCount
        //{
        //    get { return _ElementCount; }
        //}
        ///// <summary>
        ///// Gets the value as double.
        ///// </summary>
        ///// <param name="element">The element.</param>
        ///// <returns></returns>
        ///// <remarks></remarks>
        //public double GetValueAsDouble(int element)
        //{
        //    if (element < 1 || element > this.ElementCount)
        //    {
        //        throw new ArgumentOutOfRangeException("element");
        //    }

        //    return Convert.ToDouble(this[element]);

        //}
        ///// <summary>
        ///// Gets the value as single.
        ///// </summary>
        ///// <param name="element">The element.</param>
        ///// <returns></returns>
        ///// <remarks></remarks>
        //public float GetValueAsSingle(int element)
        //{
        //     if (element < 1 || element > this.ElementCount)
        //    {
        //        throw new ArgumentOutOfRangeException("element");
        //    }

        //     return Convert.ToSingle(this[element]);

        //}

        ///// <summary>
        ///// Get or set an array element values by cellNumber index.
        ///// The cellNumber index begins at 1.
        ///// </summary>
        ///// <returns></returns>
        ///// <remarks></remarks>
        //public T this[int element]
        //{
        //    get { return _Values[element - 1]; }
        //    set { _Values[element - 1] = value; }
        //}

        ///// <summary>
        ///// Values the count.
        ///// </summary>
        ///// <param name="value">The value.</param>
        ///// <returns></returns>
        ///// <remarks></remarks>
        //public int ValueCount(T value)
        //{
        //    int count = 0;
        //    for (int i = 1; i <= ElementCount; i++)
        //    {
        //        if (_gn.Equal(this[i], value)) count++;
        //    }
        //    return count;
        //}

        ///// <summary>
        ///// Set all elements of this matrix equal to default value.
        ///// </summary>
        ///// <remarks></remarks>
        //public void ClearValues()
        //{
        //    T val = default(T);
        //    for (int i = 0; i < this.ElementCount; i++)
        //    {
        //        _Values[i] = val;
        //    }
        //}

        ///// <summary>
        ///// Assign a constant constantValue to all elements of the matrix.
        ///// </summary>
        ///// <param name="value">The value.</param>
        ///// <remarks></remarks>
        //public void SetValues(T value)
        //{
        //    for (int i = 0; i < this.ElementCount; i++)
        //    { _Values[i] = value; }
        //}
        /// <summary>
        /// Set the element values of this array to the corresponding elements
        /// of the input array.
        /// </summary>
        /// <param name="a">A.</param>
        /// <remarks></remarks>
        public void SetValues(Array1d<T> a)
        {
            if (a == null)
            { throw new ArgumentNullException("a"); }
            if (a.ElementCount != ElementCount)
            { throw new ArgumentException("Dimensions of the input array do not match."); }
            this.SetValuesFromNumberArray(a);
        }

        ///// <summary>
        ///// Gets a value indicating whether this instance is constant.
        ///// </summary>
        ///// <remarks></remarks>
        //public bool IsConstant
        //{
        //    get
        //    {
        //        T val = _Values[0];
        //        for (int i = 1; i < this.ElementCount; i++)
        //        { if (!_gn.Equal(_Values[i], val)) return false; }
        //        return true;
        //    }
        //}

        ///// <summary>
        ///// Compares the specified constant with each element in the matrix and returns the
        ///// number of values that are less than, greater than, or equal to the constant.
        ///// </summary>
        ///// <param name="c">The c.</param>
        ///// <returns></returns>
        ///// <remarks></remarks>
        //public PumaGlobal.ArrayComparisonStats CompareTo(T c)
        //{
        //    return new PumaGlobal.ArrayComparisonStats();
        //}
        ///// <summary>
        ///// Compares each element in the input matrix with the corresponding element in the matrix and returns the
        ///// number of values that are less than, greater than, or equal to the input matrix values.
        ///// </summary>
        ///// <param name="a">A.</param>
        ///// <returns></returns>
        ///// <remarks></remarks>
        //public PumaGlobal.ArrayComparisonStats CompareTo(Array1d<T> a)
        //{
        //    return new PumaGlobal.ArrayComparisonStats();
        //}

        /// <summary>
        /// Gets a copy of this array.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public Array1d<T> GetCopy()
        {
            Array1d<T> a = new Array1d<T>(ElementCount);
            for (int i = 1; i <= ElementCount; i++)
            {
                a[i] = this[i];
            }
            return a;
        }
        /// <summary>
        /// Gets the copy as single.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public Array1d<float> GetCopyAsSingle()
        {
            Array1d<float> a = new Array1d<float>(this.ElementCount);
            for (int i = 1; i <= this.ElementCount; i++)
            {
                a[i] = this.GetValueAsSingle(i);
            }
            return a;
        }
        /// <summary>
        /// Gets the copy as double.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public Array1d<double> GetCopyAsDouble()
        {
            Array1d<double> a = new Array1d<double>(this.ElementCount);
            for (int i = 1; i <= this.ElementCount; i++)
            {
                a[i] = this.GetValueAsDouble(i);
            }
            return a;
        }

        /// <summary>
        /// Toes the array.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public T[] ToArray()
        {
            T[] a = new T[ElementCount];
            for (int i = 1; i <= ElementCount; i++)
            {
                a[i] = this[i];
            }
            return a;
        }

        #region Math Members
        /// <summary>
        /// Adds elements of the input matrix to the corresponding elements
        /// of this matrix.
        /// </summary>
        /// <param name="m">The m.</param>
        /// <remarks></remarks>
        public void Add(Array1d<T> m)
        {
            this.AddNumberArray(m as NumberArray<T>);
        }
        /// <summary>
        /// Subtracts the specified m.
        /// </summary>
        /// <param name="m">The m.</param>
        /// <remarks></remarks>
        public void Subtract(Array1d<T> m)
        {
            this.SubtractNumberArray(m as NumberArray<T>);
        }
        /// <summary>
        /// Multiply each element of this matrix by the corresponding element
        /// of the input matrix.
        /// </summary>
        /// <param name="m">The m.</param>
        /// <remarks></remarks>
        public void Multiply(Array1d<T> m)
        {
            this.MultiplyNumberArray(m as NumberArray<T>);
        }
        /// <summary>
        /// Divide each element of this matrix by the corresponding element
        /// of the input matrix.
        /// </summary>
        /// <param name="m">The m.</param>
        /// <remarks></remarks>
        public void Divide(Array1d<T> m)
        {
            this.DivideNumberArray(m as NumberArray<T>);
        }
        #endregion


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
                    T obj = default(T);
                    _PumaType = "Array1d" + obj.GetType().Name;
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
                    _PumaType = "Array1d" + obj.GetType().Name;
                    _DefaultName = _PumaType;
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
                System.Xml.XmlNode elementsNode;
                System.Xml.XmlNode attrNode;

                doc.LoadXml(xmlString);
                System.Xml.XmlNode root = doc.DocumentElement;

                if (root == null)
                { throw new Exception("Error loading Puma object from XML."); }

                if (!DataObjectUtility.IsValidPumaType(root, PumaType))
                { throw new Exception("Error loading Puma object from XML. Incorrect PumaType."); }

                attrNode = root.Attributes.GetNamedItem("ElementCount");
                int elements = int.Parse(attrNode.InnerText);

                elementsNode = root.SelectSingleNode("Elements");
                bool isConstant = false;
                if (elementsNode == null) isConstant = true;

                if (isConstant)
                {
                    T val;
                    node = root.SelectSingleNode("ConstantValue");
                    if (!this.Math.TryParse(node.InnerText, out val))
                    { val = default(T); }
                    CreateNewArray(elements, val);
                }
                else
                {
                    CreateNewArray(elements);

                    T[] a;
                    a = NumberArrayIO<T>.StringToArray(elementsNode.InnerText);
                    for (int i = 0; i < elements; i++)
                    { this[i + 1] = a[i]; }
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
                attNode = root.Attributes.SetNamedItem(doc.CreateAttribute("ElementCount"));
                attNode.InnerText = ElementCount.ToString();
                DataObjectUtility.AppendPumaTypeAttribute(this, root);

                if (IsConstant)
                {
                    T val = this[1];
                    root.AppendChild(doc.CreateElement("ConstantValue"));
                    root.LastChild.InnerText = val.ToString();
                }
                else
                {
                    root.AppendChild(doc.CreateElement("Elements"));
                    root.LastChild.InnerText = NumberArrayIO<T>.ArrayToString(this.BaseValues, false);
                    
                }

                return root.OuterXml;

                
            }
            catch (Exception ex)
            {

                throw;
            }
        }

        #endregion

        #region Private Methods
        /// <summary>
        /// Creates the new array.
        /// </summary>
        /// <param name="elements">The elements.</param>
        /// <remarks></remarks>
        private void CreateNewArray(int elements)
        {
            T val = default(T);
            CreateNewArray(elements, val);
        }
        /// <summary>
        /// Creates the new array.
        /// </summary>
        /// <param name="elements">The elements.</param>
        /// <param name="initialValue">The initial value.</param>
        /// <remarks></remarks>
        private void CreateNewArray(int elements, T initialValue)
        {
            this.ResizeArray(elements, initialValue);
        }
        /// <summary>
        /// Creates the new array.
        /// </summary>
        /// <param name="arrayData">The array data.</param>
        /// <remarks></remarks>
        private void CreateNewArray(Array1d<T> arrayData)
        {
            if (arrayData == null)
            { throw (new ArgumentNullException()); }
            this.ResizeArray(arrayData.BaseValues);
        }


        #endregion

    }
}
