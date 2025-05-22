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
    public class IndexRange : IDataObject
    {
        #region Fields
        #endregion

        #region Public Methods

        /// <summary>
        /// 
        /// </summary>
        private int _ToIndex = 0;
        /// <summary>
        /// Gets or sets to index.
        /// </summary>
        /// <value>To index.</value>
        /// <remarks></remarks>
        public int ToIndex
        {
            get { return _ToIndex; }
            set { _ToIndex = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        private int _FromIndex = 0;
        /// <summary>
        /// Gets or sets from index.
        /// </summary>
        /// <value>From index.</value>
        /// <remarks></remarks>
        public int FromIndex
        {
            get { return _FromIndex; }
            set { _FromIndex = value; }
        }

        /// <summary>
        /// Determines whether the index value is contained in the specified index range.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns><c>true</c> if the index is within the index range; otherwise, <c>false</c>.</returns>
        /// <remarks></remarks>
        public bool Contains(int index)
        {
            if (index < FromIndex) return false;
            if (index > ToIndex) return false;
            return true;
        }

        /// <summary>
        /// Gets the count.
        /// </summary>
        /// <remarks></remarks>
        public int Count
        {
            get
            {
                int count = ToIndex - FromIndex + 1;
                if (count < 0) count = 0;
                return count;
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
                System.Xml.XmlNode node = null;

                System.Xml.XmlDocument doc = DataObjectUtility.XmlWrapperDoc(this, elementName);
                System.Xml.XmlNode root = doc.DocumentElement;

                node = root.Attributes.SetNamedItem(doc.CreateAttribute("fromIndex"));
                node.InnerText = FromIndex.ToString();

                node = root.Attributes.SetNamedItem(doc.CreateAttribute("toIndex"));
                node.InnerText = ToIndex.ToString();

                DataObjectUtility.AppendPumaTypeAttribute(this, root);

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
