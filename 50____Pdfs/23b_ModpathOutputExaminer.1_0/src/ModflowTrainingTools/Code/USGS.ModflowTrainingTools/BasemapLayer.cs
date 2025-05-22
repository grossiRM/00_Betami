using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;
using USGS.Puma.UI.MapViewer;

namespace USGS.ModflowTrainingTools
{
    public class BasemapLayer
    {
        #region Private Fields
        private char _Delimiter = ';';
        #endregion

        #region Public Static Methods
        public static BasemapLayer Parse(string sourceString)
        {
            BasemapLayer layer = new BasemapLayer();
            if (layer.Load(sourceString))
            { return layer; }
            else
            { return null; }
        }
        public static BasemapLayer CreateFromXml(XmlElement element)
        {
            try
            {
                if (element.Name != "BasemapLayer") return null;
                BasemapLayer layer = new BasemapLayer();
                layer.LocalName = element.SelectSingleNode("LocalName").InnerText;
                layer.Description = element.SelectSingleNode("Description").InnerText;

                // extract color
                XmlNode cNode = element.SelectSingleNode("Color");
                int r = int.Parse(cNode.SelectSingleNode("R").InnerText);
                int g = int.Parse(cNode.SelectSingleNode("G").InnerText);
                int b = int.Parse(cNode.SelectSingleNode("B").InnerText);
                layer.Color = System.Drawing.Color.FromArgb(r, g, b);

                layer.Size = int.Parse(element.SelectSingleNode("Size").InnerText);
                layer.Style = int.Parse(element.SelectSingleNode("Style").InnerText);
                layer.IsVisible = bool.Parse(element.SelectSingleNode("Visible").InnerText);
                return layer;
            }
            catch (Exception)
            {
                return null;
            }
        }
        public static void AppendXmlElement(BasemapLayer layer, XmlNode xmlNode)
        {
            try
            {
                XmlElement node = null;
                XmlDocument doc = xmlNode.OwnerDocument;

                XmlNode root = doc.CreateElement("BasemapLayer");

                node = doc.CreateElement("LocalName");
                node.InnerText = layer.LocalName;
                root.AppendChild(node);

                node = doc.CreateElement("Description");
                node.InnerText = layer.Description;
                root.AppendChild(node);

                node = doc.CreateElement("Color");
                node.InnerText = layer.Color.ToString();
                root.AppendChild(node);

                node = doc.CreateElement("Size");
                node.InnerText = layer.Size.ToString();
                root.AppendChild(node);

                node = doc.CreateElement("Style");
                node.InnerText = layer.Style.ToString();
                root.AppendChild(node);

                node = doc.CreateElement("Visible");
                node.InnerText = layer.IsVisible.ToString();
                root.AppendChild(node);

                xmlNode.AppendChild(root);
                
            }
            catch (Exception)
            {
                throw;
            }
        }
        #endregion

        #region Constructors
        protected BasemapLayer()
        {

        }
        public BasemapLayer(string localFilename, System.Drawing.Color color, int size, int style,string description)
        {
            if (!Create(localFilename,color,size,style,description))
                throw new Exception("Error creating basemap layer.");
        }
        private bool Create(string localFilename, System.Drawing.Color color, int size, int style, string description)
        {
            LocalName = localFilename;
            Color = color;
            Size = size;
            Style = style;
            Description = description;
            return true;
        }
        #endregion

        #region Public Properties
        private string _LocalName = "";
        public string LocalName
        {
            get { return _LocalName; }
            set { _LocalName = value; }
        }

        private string _Description = "";
        public string Description
        {
            get { return _Description; }
            set { _Description = value; }
        }

        private System.Drawing.Color _Color = System.Drawing.Color.Black;
        public System.Drawing.Color Color
        {
            get { return _Color; }
            set { _Color = value; }
        }

        private int _Size = 1;
        public int Size
        {
            get { return _Size; }
            set { _Size = value; }
        }

        private int _Style = 0;
        public int Style
        {
            get { return _Style; }
            set { _Style = value; }
        }

        private bool _IsVisible = true;
        public bool IsVisible
        {
            get { return _IsVisible; }
            set { _IsVisible = value; }
        }

        #endregion

        #region Public Methods
        public override string ToString()
        {
            if (!string.IsNullOrEmpty(LocalName))
            {
                StringBuilder sb = new StringBuilder();
                sb.Append(LocalName).Append(_Delimiter);
                sb.Append(Description).Append(_Delimiter);
                sb.Append(Color).Append(_Delimiter);
                sb.Append(Size).Append(_Delimiter);
                sb.Append(Style);
                return sb.ToString(0, sb.Length);
            }
            else
            { return ""; }
        }


        #endregion

        #region Protected Methods
        protected bool Load(string sourceString)
        {
            string[] tokens = sourceString.Split(';');
            if (tokens.Length < 5) return false;

            _LocalName = tokens[0].Trim();
            _Description = tokens[1].Trim();
            int c = int.Parse(tokens[2]);
            _Color = MoUtilities.FromMoColor((uint)c);
            _Size = int.Parse(tokens[3]);
            _Style = int.Parse(tokens[4]);
            _IsVisible = true;
            return true;
        }
        #endregion

        #region Private Methods
        private void Reset()
        {

        }
        #endregion
    }
}
