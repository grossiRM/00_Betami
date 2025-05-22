using System.Xml.Serialization; 
using System.IO; 
using System.Xml; 
using Microsoft.VisualBasic;
using System;
using System.Collections;
using System.Data;
using System.Diagnostics;
using USGS.Puma.Core;

namespace USGS.Puma.Utilities
{
    public class DataObjectUtility  
    { 
        #region Static Public Methods
        /// <summary>
        /// 
        /// </summary>
        /// <param name="dataObject"></param>
        /// <param name="pumaType"></param>
        /// <param name="defaultName"></param>
        static public void CreatePumaTypeInfo(object dataObject, out string pumaType, out string defaultName)
        {
            pumaType = "";
            defaultName = "DataObject";

            if (dataObject == null) return;

            Type t = dataObject.GetType();
            string s = t.FullName;
            pumaType = s;

            int n = s.IndexOf('`');
            if (n < 0)
            {
                string[] parts = s.Split(new char[1] {'.'});
                defaultName = parts[parts.Length-1];
            }
            else
            {
                string sName = s.Remove(n);
                string[] parts = sName.Split(new char[1] { '.' });
                defaultName = parts[parts.Length - 1];
            }

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dataObject"></param>
        /// <param name="dataObjectNode"></param>
        static public void AppendPumaTypeAttribute(IDataObject dataObject, XmlNode dataObjectNode)
        {
            if (dataObject == null)
                throw new NullReferenceException();

            if (dataObjectNode == null)
                throw new NullReferenceException();

            XmlNode node = null;
            node = dataObjectNode.Attributes.GetNamedItem("PumaType");
            if (node == null)
                node = dataObjectNode.Attributes.SetNamedItem(dataObjectNode.OwnerDocument.CreateAttribute("PumaType"));
            
            node.InnerText = dataObject.PumaType;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="elementName"></param>
        /// <param name="pumaType"></param>
        /// <param name="version"></param>
        /// <returns></returns>
        static public XmlDocument XmlWrapperDoc(string elementName, string pumaType, int version)
        {
            string sName = null;
            XmlDocument pDOM = null;
            XmlNode pNode = null;
            XmlNode pRoot = null;

            try
            {
                if (elementName == null) return null;
                if (elementName.Trim().Length == 0) return null;

                pDOM = new XmlDocument();
                pDOM.LoadXml("<" + elementName + "/>");
                pRoot = pDOM.DocumentElement;

                if (pumaType != null)
                {
                    pNode = pRoot.Attributes.SetNamedItem(pDOM.CreateAttribute("PumaType"));
                    pNode.InnerText = pumaType;
                }

                if (version > 1)
                {
                    pNode = pRoot.Attributes.SetNamedItem(pDOM.CreateAttribute("Version"));
                    pNode.InnerText = version.ToString();
                }

                return pDOM;

            }
            catch (Exception ex)
            {
                //System.Windows.Forms.MessageBox.Show("PumaObjectHelper:" + Constants.vbNewLine + ex.Message);
                return null;
            }
            finally
            {
                pDOM = null;
                pRoot = null;
                pNode = null;
            }

        }
        static public XmlDocument XmlWrapperDoc(IDataObject PumaObj)
        {
            try
            {
                if (PumaObj == null) return null;
                return XmlWrapperDoc(PumaObj.DefaultName, null, 0);
            }
            catch (Exception ex)
            {
                //System.Windows.Forms.MessageBox.Show("PumaObjectHelper:" + Constants.vbNewLine + ex.Message);
                return null;
            }
        }
        static public XmlDocument XmlWrapperDoc(IDataObject PumaObj,string elementName)
        {
            try
            {
                if (PumaObj == null) return null;
                return XmlWrapperDoc(elementName, null, 0);

            }
            catch (Exception ex)
            {
                //System.Windows.Forms.MessageBox.Show("PumaObjectHelper:" + Constants.vbNewLine + ex.Message);
                return null;
            }
        }
        static public bool IsValidPumaType(System.Xml.XmlNode rootNode, string pumaType)
        {
            try
            {
                System.Xml.XmlNode node;
                if (rootNode == null) return false;

                try
                { node = rootNode.Attributes.GetNamedItem("pumaType"); }
                catch (Exception)
                { return true; }

                if (node == null) return true;
                return (node.InnerText == pumaType);

            }
            catch (Exception)
            {
                return false;
            }
        }
        static public object LoadFromFile(string fileName)
        {
            try
            {

                return null;
            }
            catch (Exception)
            {
                return null;
            }
        }
        static public object LoadFromString(string xmlString)
        {
            try
            {

                return null;
            }
            catch (Exception)
            {
                return null;
            }
        }
        static public void SaveToFile(string sfileName)
        {
        }
        static public void SaveToString(string xmlString)
        {
        }
        #endregion
       
        
    } 
    
    
} 
