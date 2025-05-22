using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text;
using USGS.Puma.Core;
using USGS.Puma.Utilities;
using System.Xml;
using USGS.Puma.NTS.Geometries;
using USGS.Puma.NTS.IO.GML2;

namespace USGS.Puma.Core
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public class ContourLineList : Collection<ContourLine>, IDataObject
    {
        #region Public Static Methods
        /// <summary>
        /// Creates the specified reader.
        /// </summary>
        /// <param name="reader">The reader.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public static ContourLineList Create(XmlTextReader reader)
        {
            ContourLineList list = null;

            // Check to see if the current node is the correct type.
            if (reader.Name == "ContourLineList")
            {
                list = new ContourLineList();
                GMLReader gmlReader = new GMLReader();
                
                while (reader.Read())
                {
                    switch (reader.NodeType)
                    {
                        case XmlNodeType.Element:
                            if (reader.Name == "ContourLine")
                            {
                                list.Add(ContourLine.Create(reader,gmlReader));
                            }
                            break;
                        case XmlNodeType.EndElement:
                            if (reader.Name == "ContourLineList")
                                return list;
                            break;
                        default:
                            break;
                    }
                }

                throw new Exception("Error creating ContourLineList from XML data.");

            }
            else
            { throw new ArgumentException("Incorrect data type. Expected ContourLineList data."); }

        }

        #endregion

        /// <summary>
        /// Gets the contour levels.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public float[] GetContourLevels()
        {
            Dictionary<float, object> dict = new Dictionary<float,object>();
            List<float> list = new List<float>();

            foreach (ContourLine c in this)
            {
                if (!dict.ContainsKey(c.ContourLevel))
                {
                    dict.Add(c.ContourLevel, null);
                    list.Add(c.ContourLevel);
                }
            }
            list.Sort();
            return list.ToArray();
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
                    _DefaultName = "ContourLineList";
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
                    _DefaultName = "ContourLineList";
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
                System.Xml.XmlDocument doc = new System.Xml.XmlDocument();
                System.Xml.XmlNode node;

                doc.LoadXml(xmlString);
                System.Xml.XmlNode root = doc.DocumentElement;

                if (root == null)
                { throw new Exception("Error loading Puma object from XML."); }

                if (!DataObjectUtility.IsValidPumaType(root, PumaType))
                { throw new Exception("Error loading Puma object from XML. Incorrect PumaType."); }

                this.Clear();

                ContourLine cl = null;
                System.Xml.XmlNode nodeContourLines = root.SelectSingleNode("ContourLines");
                foreach (System.Xml.XmlNode child in nodeContourLines.ChildNodes)
                {
                    cl = new ContourLine();

                    if (!cl.LoadFromXml(child.OuterXml))
                        throw new Exception("Error loading Puma ContourLine object from XML.");

                    this.Add(cl);
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
                System.Xml.XmlNode nodePoints = root.AppendChild(doc.CreateElement("ContourLines"));

                foreach (ContourLine cl in this)
                {
                    sXml = cl.SaveAsXml("ContourLine");
                    if (sXml != null)
                    {
                        docFrag = doc.CreateDocumentFragment();
                        docFrag.InnerXml = sXml;
                        nodePoints.AppendChild(docFrag);
                    }
                }

                return root.OuterXml;


            }
            catch (Exception ex)
            {
                throw new Exception("Error saving Puma object as XML.");
            }
        }

        #endregion
    }
}
