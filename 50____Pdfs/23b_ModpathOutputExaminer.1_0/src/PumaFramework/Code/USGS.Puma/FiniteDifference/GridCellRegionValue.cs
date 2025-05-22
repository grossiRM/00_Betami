using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Core;
using USGS.Puma.Utilities;

namespace USGS.Puma.FiniteDifference
{
    public class GridCellRegionValue : GridCellRegion
    {
        #region Fields
        private double mDataValue = 0;
        #endregion

        #region Constructors
        public GridCellRegionValue()
        {
            IsValid = false;

            FromCell = new GridCell();
            ToCell = new GridCell();
            DataValue = 0;

            IsValid = true;

        }
        public GridCellRegionValue(GridCell fromCell, GridCell toCell, double dataValue)
        {
            IsValid = false;

            FromCell = new GridCell();
            ToCell = new GridCell();
            if (fromCell == null) return;
            if (toCell == null) return;

            if (!FromCell.LoadFromXml(fromCell.SaveAsXml())) return;
            if (!ToCell.LoadFromXml(toCell.SaveAsXml())) return;

            DataValue = dataValue;

            IsValid = true;

        }


        #endregion

        #region Public Methods
        public double DataValue
        {
            get { return mDataValue; }
            set { mDataValue = value; }
        }
        #endregion

        #region IDataObject Members

        private string _PumaType = "";
        /// <summary>
        /// Gets the fully qualified type name of this object.
        /// </summary>
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

        private string _DefaultName = "";
        /// <summary>
        /// Gets the default name that will be used for the root XML element of this
        /// class.
        /// </summary>
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
        public int Version
        { get { return 1; } }

        /// <summary>
        ///  Returns True if the DataObject is properly initialized.
        /// </summary>
        private bool m_IsValid = false;
        public bool IsValid
        {
            get { return m_IsValid; }
            private set { m_IsValid = value; }
        }


        #endregion 

        #region ISerializeXml Members
        public bool LoadFromXml(string xmlString)
        {
            try
            {
                IsValid = false;

                System.Xml.XmlDocument doc = new System.Xml.XmlDocument();
                System.Xml.XmlNode node;

                doc.LoadXml(xmlString);
                System.Xml.XmlNode root = doc.DocumentElement;

                if (root == null)
                { throw new Exception("Error loading Puma object from XML."); }

                if (!DataObjectUtility.IsValidPumaType(root, PumaType))
                { throw new Exception("Error loading Puma object from XML. Incorrect PumaType."); }

                /// Add code here
                FromCell = new GridCell();
                node = root.SelectSingleNode("FromCell");
                if (!FromCell.LoadFromXml(node.OuterXml))
                { throw new Exception("Error loading Puma object from XML."); }

                ToCell = new GridCell();
                node = root.SelectSingleNode("ToCell");
                if (!ToCell.LoadFromXml(node.OuterXml))
                { throw new Exception("Error loading Puma object from XML."); }

                node = root.SelectSingleNode("DataValue");
                DataValue = double.Parse(node.InnerText);

                IsValid = true;
                return true;

            }
            catch (Exception)
            {
                IsValid = false;
                return false;
            }
        }
        public string SaveAsXml()
        { return SaveAsXml(DefaultName); }
        public string SaveAsXml(string elementName)
        {
            try
            {
                System.Xml.XmlNode node = null;

                System.Xml.XmlDocument doc = DataObjectUtility.XmlWrapperDoc(this, elementName);
                System.Xml.XmlNode root = doc.DocumentElement;

                DataObjectUtility.AppendPumaTypeAttribute(this, root);

                System.Xml.XmlDocumentFragment docFrag = doc.CreateDocumentFragment();
                docFrag.InnerXml = FromCell.SaveAsXml();
                node = root.AppendChild(docFrag);

                docFrag = doc.CreateDocumentFragment();
                docFrag.InnerXml = ToCell.SaveAsXml();
                node = root.AppendChild(docFrag);

                node = root.AppendChild(doc.CreateElement("DataValue"));
                node.InnerText = DataValue.ToString();

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
