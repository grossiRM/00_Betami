using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace USGS.Puma.FiniteDifference
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public class GridGeoReference
    {        
        #region Public Static Methods
        /// <summary>
        /// Reads the specified file.
        /// </summary>
        /// <param name="filename">The name of a Modflow metadata file.</param>
        /// <returns>An instance of the <see cref="GridGeoReference"></see> class.</returns>
        /// <remarks></remarks>
        public static GridGeoReference Read(string filename)
        {
            try
            {
                XmlDocument doc = new XmlDocument();
                doc.Load(filename);
                //if (doc.DocumentElement.Name != "Metadata")
                //    throw new Exception("The specified file is not an HDViewer definition file.");

                XmlNode root = doc.DocumentElement.SelectSingleNode("GridGeoReference");
                if (root.Name != "GridGeoReference")
                    throw new Exception("The specified file is not an GridGeoReference file.");

                GridGeoReference gridGeoRef = null;
                if (root != null)
                {
                    gridGeoRef = CreateFromXml(root);
                }
                return gridGeoRef;

            }
            catch (Exception)
            {
                throw new Exception("Error reading GridGeoReference file.");
            }
            
        }
        /// <summary>
        /// Creates an instance of the <see cref="GridGeoReference"></see> class from XML.
        /// </summary>
        /// <param name="xmlElement">The XML element.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public static GridGeoReference CreateFromXml(XmlNode xmlElement)
        {
            if (xmlElement.Name != "GridGeoReference")
                return null;

            GridGeoReference gridGeoRef = new GridGeoReference();

            XmlNode root = xmlElement;
            XmlNode node2 = null;

            node2 = root.SelectSingleNode("OriginX");
            gridGeoRef.OriginX = double.Parse(node2.InnerText);
            node2 = root.SelectSingleNode("OriginY");
            gridGeoRef.OriginY = double.Parse(node2.InnerText);
            node2 = root.SelectSingleNode("Angle");
            gridGeoRef.Angle = double.Parse(node2.InnerText);
            node2 = root.SelectSingleNode("ProjectionString");
            gridGeoRef.ProjectionString = node2.InnerText;

            return gridGeoRef;

        }
        /// <summary>
        /// Writes the specified Modflow metadata to a file.
        /// </summary>
        /// <param name="filename">The name of the Modflow metadata file.</param>
        /// <param name="gridGeoRef">The .</param>
        /// <remarks></remarks>
        public static void Write(string filename, GridGeoReference gridGeoRef)
        {
            using (XmlTextWriter writer = new XmlTextWriter(filename, null))
            {
                writer.Formatting = Formatting.Indented;
                writer.Indentation = 2;

                // Write processing instruction header
                char quote = '"';
                string header = "version=" + quote + "1.0" + quote + " encoding=" + quote + "utf-8" + quote;
                writer.WriteProcessingInstruction("xml", header);

                // Write root element start tag.
                writer.WriteStartElement("Metadata");
                Write(writer, gridGeoRef);
                writer.WriteEndElement();
                
            }
        }
        /// <summary>
        /// Writes the specified writer.
        /// </summary>
        /// <param name="writer">The writer.</param>
        /// <param name="gridGeoRef">The grid geo ref.</param>
        /// <remarks></remarks>
        public static void Write(XmlTextWriter writer, GridGeoReference gridGeoRef)
        {
            writer.WriteStartElement("GridGeoReference");
            writer.WriteElementString("OriginX", gridGeoRef.OriginX.ToString());
            writer.WriteElementString("OriginY", gridGeoRef.OriginY.ToString());
            writer.WriteElementString("Angle", gridGeoRef.Angle.ToString());
            writer.WriteElementString("ProjectionString", gridGeoRef.ProjectionString);
            writer.WriteEndElement();
        }
        #endregion

        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="T:System.Object"/> class.
        /// </summary>
        /// <remarks></remarks>
        public GridGeoReference()
        {
            OriginX = 0;
            OriginY = 0;
            Angle = 0;
            ProjectionString = "";
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="GridGeoReference"/> class.
        /// </summary>
        /// <param name="originX">The origin X.</param>
        /// <param name="originY">The origin Y.</param>
        /// <param name="angle">The angle.</param>
        /// <param name="projectionString">The projection string.</param>
        /// <remarks></remarks>
        public GridGeoReference(double originX, double originY, double angle, string projectionString)
        {
            OriginX = originX;
            OriginY = originY;
            Angle = angle;
            ProjectionString = projectionString;
        }
        #endregion

        #region Public Properties
        /// <summary>
        /// 
        /// </summary>
        private double _OriginX = 0;
        /// <summary>
        /// Gets or sets the origin X.
        /// </summary>
        /// <value>The origin X.</value>
        /// <remarks></remarks>
        public double OriginX
        {
            get { return _OriginX; }
            set { _OriginX = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        private double _OriginY = 0;
        /// <summary>
        /// Gets or sets the origin Y.
        /// </summary>
        /// <value>The origin Y.</value>
        /// <remarks></remarks>
        public double OriginY
        {
            get { return _OriginY; }
            set { _OriginY = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        private double _Angle = 0;
        /// <summary>
        /// Gets or sets the angle.
        /// </summary>
        /// <value>The angle.</value>
        /// <remarks></remarks>
        public double Angle
        {
            get { return _Angle; }
            set { _Angle = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        private string _ProjectionString = "";
        /// <summary>
        /// Gets or sets the projection string.
        /// </summary>
        /// <value>The projection string.</value>
        /// <remarks></remarks>
        public string ProjectionString
        {
            get { return _ProjectionString; }
            set { _ProjectionString = value; }
        }
        #endregion

        #region Public Methods
        /// <summary>
        /// 
        /// </summary>
        /// <param name="baselinePointX"></param>
        /// <param name="baselinePointY"></param>
        public void ComputeAndAssignAngle(double baselinePointX, double baselinePointY)
        {
            double a = ComputeAngle(baselinePointX, baselinePointY);
            Angle = a;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="baselinePointX"></param>
        /// <param name="baselinePointY"></param>
        /// <returns></returns>
        public double ComputeAngle(double baselinePointX, double baselinePointY)
        {
            double dy = (baselinePointY - OriginY);
            double dx = (baselinePointX - OriginX);
            double r = Math.Atan2(dy, dx);
            double a = r * (180 / Math.PI);
            return a;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="originX"></param>
        /// <param name="originY"></param>
        /// <param name="baselinePointX"></param>
        /// <param name="baselinePointY"></param>
        /// <returns></returns>
        public double ComputeAngle(double originX, double originY, double baselinePointX, double baselinePointY)
        {
            double dy = (baselinePointY - originY);
            double dx = (baselinePointX - originX);
            double r = Math.Atan2(dy, dx);
            double a = r* (180 / Math.PI);
            return a;

        }
        #endregion

    }
}
