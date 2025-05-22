using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Core;
using USGS.Puma.Utilities;

namespace USGS.Puma.FiniteDifference
{
    public class GridCell : IDataObject, IGridCell
    {
        #region Fields
        private bool mHasGrid;
        private bool mHasLayer;
        private int _Grid;
        private int _Layer;
        private int mRow;
        private int mColumn;
        #endregion

        #region Constructors
        public GridCell()
        {
            try
            {
                Grid = 0;
                Layer = 0;
                Row = 0;
                Column = 0;
                IsValid = true;
            }
            catch (Exception)
            {
                IsValid = false;
            }
        }
        public GridCell(int row, int column)
        {
            Grid = 0;
            Layer = 0;
            Row = row;
            Column = column;
            IsValid = true;
        }
        public GridCell(int layer, int row, int column)
        {
            Grid = 0;
            Layer = layer;
            Row = row;
            Column = column;
            IsValid = true;
        }
        public GridCell(int grid, int layer, int row, int column)
        {
            Grid = grid;
            Layer = layer;
            Row = row;
            Column = column;
            IsValid = true;
        }
        #endregion

        #region Public Methods
        public int Grid
        {
            get
            { return _Grid; }

            set
            { _Grid = value; }
        }
        public int Layer
        {
            get
            { return _Layer; }
            set
            { _Layer = value; }
        }
        public int Row
        {
            get { return mRow; }
            set { mRow = value; }
        }
        public int Column
        {
            get { return mColumn; }
            set { mColumn = value; }
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
                Row = 0;
                Column = 0;
                Grid = 0;
                Layer = 0;

                System.Xml.XmlDocument doc = new System.Xml.XmlDocument();
                System.Xml.XmlNode node;

                doc.LoadXml(xmlString);
                System.Xml.XmlNode root = doc.DocumentElement;

                if (root == null)
                { throw new Exception("Error loading Puma object from XML."); }

                if (!DataObjectUtility.IsValidPumaType(root, PumaType))
                { throw new Exception("Error loading Puma object from XML. Incorrect PumaType."); }

                /// Add code here
                node = root.Attributes.GetNamedItem("G");
                if (node != null)
                { Grid = int.Parse(node.InnerText); }

                node = root.Attributes.GetNamedItem("L");
                if (node != null)
                { Layer = int.Parse(node.InnerText); }

                node = root.Attributes.GetNamedItem("R");
                Row = int.Parse(node.InnerText);

                node = root.Attributes.GetNamedItem("C");
                Column = int.Parse(node.InnerText);

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
        {
            return SaveAsXml(DefaultName);
        }

        public string SaveAsXml(string elementName)
        {
            try
            {
                System.Xml.XmlNode node = null;

                System.Xml.XmlDocument doc = DataObjectUtility.XmlWrapperDoc(this, elementName);
                System.Xml.XmlNode root = doc.DocumentElement;

                if (Grid != 0)
                {
                    node = root.Attributes.SetNamedItem(doc.CreateAttribute("G"));
                    node.InnerText = Grid.ToString();
                }

                if (Layer != 0)
                {
                    node = root.Attributes.SetNamedItem(doc.CreateAttribute("L"));
                    node.InnerText = Layer.ToString();
                }

                node = root.Attributes.SetNamedItem(doc.CreateAttribute("R"));
                node.InnerText = Row.ToString();

                node = root.Attributes.SetNamedItem(doc.CreateAttribute("C"));
                node.InnerText = Column.ToString();

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
