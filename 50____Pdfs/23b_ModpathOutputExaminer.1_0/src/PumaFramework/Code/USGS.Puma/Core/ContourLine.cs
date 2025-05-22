using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Core;
using USGS.Puma.Utilities;
using USGS.Puma.NTS.IO.GML2;
using USGS.Puma.NTS.Geometries;
using GeoAPI.Geometries;
using System.Xml;

namespace USGS.Puma.Core
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public class ContourLine : IDataObject
    {
        #region Public Static Methods

        /// <summary>
        /// Creates the specified reader.
        /// </summary>
        /// <param name="reader">The reader.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public static ContourLine Create(XmlTextReader reader)
        {
            GMLReader gmlReader = new GMLReader();
            return Create(reader, gmlReader);
        }
        /// <summary>
        /// Creates the specified reader.
        /// </summary>
        /// <param name="reader">The reader.</param>
        /// <param name="gmlReader">The GML reader.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public static ContourLine Create(XmlTextReader reader, GMLReader gmlReader)
        {
            if (reader.Name == "ContourLine")
            {
                try
                {
                    float level = float.Parse(reader.ReadElementString("ContourLevel"));
                    reader.Read();
                    IMultiLineString geometry = gmlReader.Read(reader) as IMultiLineString;
                    reader.Read();
                    return new ContourLine(geometry, level);
                }
                catch (Exception)
                {
                    throw new Exception("Error creating ContourLine from XML data.");
                }
            }
            else
            { throw new ArgumentException("Incorrect data type. Expected ContourLine data."); }

        }

        #endregion

        #region Fields

        #endregion

        /// <summary>
        /// Initializes a new instance of the <see cref="T:System.Object"/> class.
        /// </summary>
        /// <remarks></remarks>
        public ContourLine() : base()
        {
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="ContourLine"/> class.
        /// </summary>
        /// <param name="contour">The contour.</param>
        /// <param name="contourLevel">The contour level.</param>
        /// <remarks></remarks>
        public ContourLine(List<ICoordinate> contour, float contourLevel)
        {
            
            ICoordinate[] pts = contour.ToArray();
            ILineString ls = (ILineString)(new LineString(pts));
            ILineString[] arrayLs = new ILineString[1] { ls };
            IMultiLineString mls = (IMultiLineString)(new MultiLineString(arrayLs));
            ContourLevel = contourLevel;
            Contour = mls;
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="ContourLine"/> class.
        /// </summary>
        /// <param name="contour">The contour.</param>
        /// <param name="contourLevel">The contour level.</param>
        /// <remarks></remarks>
        public ContourLine(ILineString contour, float contourLevel)
        {
            ILineString[] arrayLs = new ILineString[1] { contour };
            IMultiLineString mls = (IMultiLineString)(new MultiLineString(arrayLs));
            ContourLevel = contourLevel;
            Contour = mls;
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="ContourLine"/> class.
        /// </summary>
        /// <param name="contour">The contour.</param>
        /// <param name="contourLevel">The contour level.</param>
        /// <remarks></remarks>
        public ContourLine(IMultiLineString contour, float contourLevel)
        {
            ContourLevel = contourLevel;
            Contour = contour;
        }

        /// <summary>
        /// 
        /// </summary>
        private float _ContourLevel;
        /// <summary>
        /// Gets or sets the contour level.
        /// </summary>
        /// <value>The contour level.</value>
        /// <remarks></remarks>
        public float ContourLevel
        {
            get { return _ContourLevel; }
            set { _ContourLevel = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        private IMultiLineString _Contour;
        /// <summary>
        /// Gets or sets the contour.
        /// </summary>
        /// <value>The contour.</value>
        /// <remarks></remarks>
        public IMultiLineString Contour
        {
            get { return _Contour; }
            set { _Contour = value; }
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
                    _DefaultName = "ContourLine";
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
                    _DefaultName = "ContourLine";
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

                node = root.SelectSingleNode("ContourLevel");
                ContourLevel = Convert.ToSingle(node.InnerText);

                node = root.SelectSingleNode("Contour");

                // ToDo -- fix this to work with the GeoAPI & NTS geometry libs
                // 
                //_Contour = new XyLineString();
                //if (! _Contour.LoadFromXml(node.OuterXml) )
                //    throw new Exception("Error loading contour line contour.");

                IsValid = true;
                return true;

            }
            catch (Exception ex)
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

                root.AppendChild(doc.CreateElement("ContourLevel"));
                root.LastChild.InnerText = ContourLevel.ToString();

                string sXml;
                System.Xml.XmlDocumentFragment docFrag;

                // ToDo -- fix this to work with the GeoAPI and NTS geomtry libs
                // 
                //sXml = Contour.SaveAsXml("Contour");
                //docFrag = doc.CreateDocumentFragment();
                //docFrag.InnerXml = sXml;
                //node = root.AppendChild(docFrag);

                return root.OuterXml;

            }
            catch (Exception ex)
            {
                throw;
            }
        }

        #endregion
    }
}
