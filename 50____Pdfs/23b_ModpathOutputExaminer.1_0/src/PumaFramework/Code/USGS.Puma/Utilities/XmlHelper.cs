using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Xml;
using USGS.Puma.Core;

namespace USGS.Puma.Utilities
{
    /// <summary>
    /// 
    /// </summary>
    public class XmlUtility
    {
        #region Static Public Methods
        /// <summary>
        /// 
        /// </summary>
        /// <param name="writer"></param>
        /// <returns></returns>
        static public XmlWriter OpenXmlWriter(StreamWriter writer)
        {
            if (writer == null)
                throw new NullReferenceException();

            XmlTextWriter xmlWriter = new XmlTextWriter(writer);
            xmlWriter.Formatting = System.Xml.Formatting.Indented;
            xmlWriter.Indentation = 2;
            return xmlWriter;

        }

        #region SaveToFile method overloads
        /// <summary>
        /// 
        /// </summary>
        /// <param name="xmlString"></param>
        /// <param name="filename"></param>
        static public void SaveToFile(string xmlString,string filename)
        {
            // Make sure the file gets deleted if it already exists. This
            // should not be necessary, but sometimes the file is not
            // updated properly if it is a preexisting file.
            System.IO.FileInfo fileInfo = new FileInfo(filename);
            if (fileInfo.Exists) fileInfo.Delete();

            using (StreamWriter writer = new StreamWriter(filename))
            {
                using(XmlTextWriter xmlWriter = new XmlTextWriter(writer))
                {
                    xmlWriter.Formatting = System.Xml.Formatting.Indented;
                    xmlWriter.Indentation = 2;
                    XmlDocument pDOM = new XmlDocument();
                    pDOM.LoadXml(xmlString);
                    pDOM.WriteTo(xmlWriter);
                    pDOM = null;
                    xmlWriter.Close();
                }
                writer.Close();
            }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="xmlDocFragment"></param>
        /// <param name="fileName"></param>
        static public void SaveToFile(System.Xml.XmlDocumentFragment xmlDocFragment, string fileName)
        {
            using(StreamWriter writer = new StreamWriter(fileName))
            {
                using(XmlTextWriter xmlWriter = new XmlTextWriter(writer))
                {
                    xmlWriter.Formatting = System.Xml.Formatting.Indented;
                    xmlWriter.Indentation = 2;
                    xmlDocFragment.WriteTo(xmlWriter);
                    xmlWriter.Close();
                }
                writer.Close();
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="xmlDoc"></param>
        /// <param name="fileName"></param>
        static public void SaveToFile(System.Xml.XmlDocument xmlDoc, string fileName)
        {
            using(StreamWriter writer = new StreamWriter(fileName))
            {
                using (XmlTextWriter xmlWriter = new XmlTextWriter(writer))
                {
                    xmlWriter.Formatting = System.Xml.Formatting.Indented;
                    xmlWriter.Indentation = 2;
                    xmlDoc.WriteTo(xmlWriter);
                    xmlWriter.Close();
                }
                writer.Close();
            }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="xmlNode"></param>
        /// <param name="fileName"></param>
        static public void SaveToFile(System.Xml.XmlNode xmlNode, string fileName)
        {
            using (StreamWriter writer = new StreamWriter(fileName))
            {
                using (XmlTextWriter xmlWriter = new XmlTextWriter(writer))
                {
                    xmlWriter.Formatting = System.Xml.Formatting.Indented;
                    xmlWriter.Indentation = 2;
                    xmlNode.WriteTo(xmlWriter);
                    xmlWriter.Close();
                }
                writer.Close();
            }

        }
        #endregion

        #endregion
    }
}
