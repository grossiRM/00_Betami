using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Utilities;

namespace USGS.Puma.Core
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public class IndexRangeValue : IndexRange
    {
        #region Fields
        /// <summary>
        /// 
        /// </summary>
        private double _DataValue = 0;
        #endregion Fields

        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="T:System.Object"/> class.
        /// </summary>
        /// <remarks></remarks>
        public IndexRangeValue()
        {
            FromIndex = 0;
            ToIndex = 0;
            DataValue = 0;
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="IndexRangeValue"/> class.
        /// </summary>
        /// <param name="fromIndex">From index.</param>
        /// <param name="toIndex">To index.</param>
        /// <param name="dataValue">The data value.</param>
        /// <remarks></remarks>
        public IndexRangeValue(int fromIndex, int toIndex, double dataValue)
        {
            FromIndex = fromIndex;
            ToIndex = toIndex;
            DataValue = dataValue;
        }
        #endregion

        #region Public Members
        /// <summary>
        /// Gets or sets the data value.
        /// </summary>
        /// <value>The data value.</value>
        /// <remarks></remarks>
        public double DataValue
        {
            get { return _DataValue; }
            set { _DataValue = value; }
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
                    string sPumaType;
                    string sDefaultName;
                    DataObjectUtility.CreatePumaTypeInfo(this, out sPumaType, out sDefaultName);
                    _PumaType = sPumaType;
                    _DefaultName = sDefaultName;
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
                    string sPumaType;
                    string sDefaultName;
                    DataObjectUtility.CreatePumaTypeInfo(this, out sPumaType, out sDefaultName);
                    _PumaType = sPumaType;
                    _DefaultName = sDefaultName;
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
                FromIndex = 0;
                ToIndex = 0;

                System.Xml.XmlDocument doc = new System.Xml.XmlDocument();
                System.Xml.XmlNode node;

                doc.LoadXml(xmlString);
                System.Xml.XmlNode root = doc.DocumentElement;

                if (root == null)
                { throw new Exception("Error loading Puma object from XML."); }

                if (!DataObjectUtility.IsValidPumaType(root, PumaType))
                { throw new Exception("Error loading Puma object from XML. Incorrect PumaType."); }

                /// Add code here
                node = root.Attributes.GetNamedItem("fromIndex");
                FromIndex = int.Parse(node.InnerText);

                node = root.Attributes.GetNamedItem("toIndex");
                ToIndex = int.Parse(node.InnerText);

                DataValue = double.Parse(root.InnerText);

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
        { return SaveAsXml(DefaultName); }
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

                node = root.Attributes.SetNamedItem(doc.CreateAttribute("fromIndex"));
                node.InnerText = FromIndex.ToString();

                node = root.Attributes.SetNamedItem(doc.CreateAttribute("toIndex"));
                node.InnerText = ToIndex.ToString();

                DataObjectUtility.AppendPumaTypeAttribute(this, root);

                root.InnerText = DataValue.ToString();

                return root.OuterXml;

            }
            catch (Exception)
            {
                throw new Exception("Error saving Puma object as XML.");
            }
        }
        #endregion

    }

}
