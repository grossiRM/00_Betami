using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text;
using USGS.Puma.Utilities;

namespace USGS.Puma.Core
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public class IndexRangeValueList : Collection<IndexRangeValue>, IDataObject
    {
        #region Fields
        #endregion

        /// <summary>
        /// 
        /// </summary>
        private string _ItemElementName = null;
        /// <summary>
        /// Gets or sets the name of the item element.
        /// </summary>
        /// <value>The name of the item element.</value>
        /// <remarks></remarks>
        public string ItemElementName
        {
            get { return _ItemElementName; }
            set { _ItemElementName = value; }
        }

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
                // clear the list
                this.Clear();

                System.Xml.XmlDocument doc = new System.Xml.XmlDocument();
                System.Xml.XmlNode node;

                doc.LoadXml(xmlString);
                System.Xml.XmlNode root = doc.DocumentElement;

                if (root == null)
                { throw new Exception("Error loading Puma object from XML."); }

                if (!DataObjectUtility.IsValidPumaType(root, PumaType))
                { throw new Exception("Error loading Puma object from XML. Incorrect PumaType."); }

                /// Add code here

                IndexRangeValue region = null;
                foreach (System.Xml.XmlNode child in root.ChildNodes)
                {
                    region = new IndexRangeValue();
                    if (region.LoadFromXml(child.OuterXml))
                    { this.Add(region); }
                }

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

                DataObjectUtility.AppendPumaTypeAttribute(this, root);

                string sXml;
                System.Xml.XmlDocumentFragment docFrag;


                foreach (IndexRangeValue region in this)
                {
                    if (ItemElementName == null)
                    { sXml = region.SaveAsXml(); }
                    else
                    { sXml = region.SaveAsXml(ItemElementName); }
                    if (sXml != null)
                    {
                        docFrag = doc.CreateDocumentFragment();
                        docFrag.InnerXml = sXml;
                        root.AppendChild(docFrag);
                    }
                }

                return root.OuterXml;

            }
            catch (Exception)
            {
                throw new Exception("Error saving Puma object as XML.");
            }

        }

        #endregion

        /// <summary>
        /// Saves as rle string.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public string SaveAsRleString()
        {
            try
            {
                StringBuilder sb = new StringBuilder();
                foreach (IndexRangeValue item in this)
                {
                    if (item.Count > 0)
                    {
                        if (item.Count == 1)
                        { sb.Append(item.DataValue).Append(","); }
                        else
                        { sb.Append(item.Count).Append(":").Append(item.DataValue).Append(","); }
                    }
                }

                return sb.ToString(0, sb.Length - 1);

            }
            catch (Exception)
            {
                return "";
            }

        }
    }

}
