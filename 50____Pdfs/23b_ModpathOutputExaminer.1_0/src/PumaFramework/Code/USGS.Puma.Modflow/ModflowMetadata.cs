using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.FiniteDifference;
using System.Xml;
using System.IO;

namespace USGS.Puma.Modflow
{
    public class ModflowMetadata
    {
        #region Static Methods
        /// <summary>
        /// 
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public static ModflowMetadata Read(string filename)
        {
            try
            {
                XmlDocument doc = new XmlDocument();
                doc.Load(filename);
                XmlNode root = doc.DocumentElement;

                if (root.Name != "ModflowMetadata")
                    throw new Exception("The specified file is not a Modflow metadata file.");

                ModflowMetadata metadata = CreateFromXml(root);
                metadata.SourcefileDirectory = Path.GetDirectoryName(filename);
                metadata.Filename = filename;
                
                return metadata;
            }
            catch (Exception)
            {
                throw new Exception("Error reading Modflow metadata file.");
            }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="filename"></param>
        /// <param name="metadata"></param>
        public static void Write(string filename, ModflowMetadata metadata)
        {
            if (metadata == null)
                throw new ArgumentNullException("The specified metadata object does not exist.");

            using (XmlTextWriter writer = new XmlTextWriter(filename, null))
            {
                writer.Formatting = Formatting.Indented;
                writer.Indentation = 2;

                // Write processing instruction header
                char quote = '"';
                string header = "version=" + quote + "1.0" + quote + " encoding=" + quote + "utf-8" + quote;
                writer.WriteProcessingInstruction("xml", header);

                // Write root element start tag.
                writer.WriteStartElement("ModflowMetadata");

                // Write GridGeoReference tag
                if (metadata.GridGeoReference != null)
                {
                    GridGeoReference.Write(writer, metadata.GridGeoReference);
                }

                // Write BasemapFile tag
                writer.WriteStartElement("BasemapFile");
                writer.WriteString(metadata.BasemapFile);
                writer.WriteEndElement();

                // Write end of element
                writer.WriteEndElement();

            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="xmlElement"></param>
        /// <returns></returns>
        public static ModflowMetadata CreateFromXml(XmlNode xmlElement)
        {
            if (xmlElement.Name != "ModflowMetadata")
                return null;

            ModflowMetadata metadata = new ModflowMetadata();
            XmlNode node = null;
            XmlNode node2 = null;
            node = xmlElement.SelectSingleNode("GridGeoReference");
            if (node != null)
            {
                GridGeoReference gridGeoRef = GridGeoReference.CreateFromXml(node);
                metadata.GridGeoReference = gridGeoRef;
            }

            node = xmlElement.SelectSingleNode("BasemapFile");
            if (node != null)
            {
                metadata.BasemapFile = node.InnerText;
            }

            return metadata;

        }

        #endregion

        public ModflowMetadata()
        {
            GridGeoReference = new GridGeoReference();
            BasemapFile = "";
            SourcefileDirectory = "";
            Filename = "";
        }

        #region Public Properties
        private USGS.Puma.FiniteDifference.GridGeoReference _GridGeoReference = null;
        /// <summary>
        /// 
        /// </summary>
        public USGS.Puma.FiniteDifference.GridGeoReference GridGeoReference
        {
            get { return _GridGeoReference; }
            set { _GridGeoReference = value; }
        }

        private string _BasemapFile = "";
        /// <summary>
        /// 
        /// </summary>
        public string BasemapFile
        {
            get { return _BasemapFile; }
            set { _BasemapFile = value; }
        }

        private string _SourcefileDirectory;
        /// <summary>
        /// 
        /// </summary>
        public string SourcefileDirectory
        {
            get { return _SourcefileDirectory; }
            set { _SourcefileDirectory = value; }
        }

        private string _Filename;
        /// <summary>
        /// 
        /// </summary>
        public string Filename
        {
            get { return _Filename; }
            set { _Filename = value; }
        }


        #endregion
    }
}
