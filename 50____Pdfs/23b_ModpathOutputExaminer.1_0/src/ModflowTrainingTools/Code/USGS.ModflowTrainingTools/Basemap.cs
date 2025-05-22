using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;
using System.IO;
using USGS.Puma.NTS.Geometries;
using USGS.Puma.NTS.Features;
using GeoAPI.Geometries;
using USGS.Puma.UI.MapViewer;

namespace USGS.ModflowTrainingTools
{
    public class Basemap
    {
        #region Public Static Methods
        public static Basemap Read(string basemapDirectory)
        {
            return Read(basemapDirectory, "basemap.pbm");
        }
        public static Basemap Read(string basemapDirectory, string localFilename)
        {
            string basemapFilename = Path.Combine(basemapDirectory, localFilename);
            if (System.IO.File.Exists(basemapFilename))
            {
                try
                {
                    XmlDocument doc = new XmlDocument();
                    doc.Load(basemapFilename);
                    XmlElement element = doc.DocumentElement;
                    if (element.Name != "Basemap") return null;

                    // Create the Basemap object
                    Basemap bm = Basemap.CreateFromXml(element, basemapDirectory, localFilename);
                    return bm;
                }
                catch (Exception)
                {
                    throw new Exception("Error reading basemap file.");
                }
            }
            else
            { throw new Exception("Basemap file does not exist."); }

        }
        public static Basemap CreateFromXml(XmlElement basemapElement, string basemapDirectory, string localFilename)
        {
           BasemapLayer layer = null;
           if (basemapElement.Name != "Basemap") return null;

            // Create the Basemap object
            Basemap bm = new Basemap(basemapDirectory, localFilename, null);

            // Create the basemap layers
            XmlNodeList nodes = basemapElement.SelectNodes("BasemapLayer");
            for (int i = 0; i < nodes.Count; i++)
            {
                layer = BasemapLayer.CreateFromXml(nodes[i] as XmlElement);
                if (layer != null)
                { bm.AddLayer(layer); }
            }
            return bm;
        }
        public static void Write(Basemap basemap)
        {
            string name = "basemap.xml";
            if (!string.IsNullOrEmpty(basemap.Filename))
                name = basemap.Filename;
            string bmFilename = Path.Combine(basemap.BasemapDirectory, name);
            Write(bmFilename, basemap);
        }
        public static void Write(string filename, Basemap basemap)
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
                writer.WriteStartElement("Basemap");

                // Write basemap layers
                for (int i = 0; i < basemap.LayerCount; i++)
                {
                    writer.WriteStartElement("BasemapLayer");
                    writer.WriteElementString("LocalName", basemap[i].LocalName);
                    writer.WriteElementString("Description", basemap[i].Description);

                    writer.WriteStartElement("Color");
                    writer.WriteElementString("R", basemap[i].Color.R.ToString());
                    writer.WriteElementString("G", basemap[i].Color.G.ToString());
                    writer.WriteElementString("B", basemap[i].Color.B.ToString());
                    writer.WriteEndElement();

                    writer.WriteElementString("Size", basemap[i].Size.ToString());
                    writer.WriteElementString("Style", basemap[i].Style.ToString());
                    writer.WriteElementString("Visible", basemap[i].IsVisible.ToString());
                    writer.WriteEndElement();
                }

                // Write root element end tag.
                writer.WriteEndElement();
        
            }
        }
        #endregion

        #region Private Fields
        private List<BasemapLayer> _Layers = null;
        #endregion

        #region Constructors
        public Basemap()
        {
            _BasemapDirectory = "";
            _Filename = "";
            _Layers = new List<BasemapLayer>();
        }
        public Basemap(string basemapDirectory)
            : this()
        {
            _BasemapDirectory = basemapDirectory;
        }
        public Basemap(string basemapDirectory, List<BasemapLayer> layers) : this()
        {
            _BasemapDirectory = basemapDirectory;

            if (layers != null)
            { _Layers = layers; }
        }
        public Basemap(string basemapDirectory, string localFilename, List<BasemapLayer> layers)
            : this()
        {
            _BasemapDirectory = basemapDirectory;
            if (!string.IsNullOrEmpty(localFilename))
                _Filename = localFilename;

            if (layers != null)
            { _Layers = layers; }

        }
        #endregion

        #region Public Properties
        private string _Filename = "";
        public string Filename
        {
            get { return _Filename; }
            set { _Filename = value; }
        }

        private string _BasemapDirectory = "";
        public string BasemapDirectory
        {
            get { return _BasemapDirectory; }
            set { _BasemapDirectory = value; }
        }

        public int LayerCount
        {
            get { return _Layers.Count; }
        }

        #endregion

        #region Public Methods
        public BasemapLayer this[int index]
        {
            get { return _Layers[index]; }
        }
        public int FindIndex(string localName)
        {
            int index = -1;
            string s = localName.ToLower();
            for (int i=0; i<_Layers.Count; i++)
            {
                if (_Layers[i].LocalName.ToLower() == s)
                {
                    index = i;
                    break;
                }
            }
            return index;
        }
        public void AddLayer(BasemapLayer layer)
        {
            if (FindIndex(layer.LocalName) < 0)
            {
                _Layers.Add(layer);
            }
        }
        public void InsertLayer(int index, BasemapLayer layer)
        {
            if (FindIndex(layer.LocalName) < 0)
            {
                _Layers.Insert(index, layer);
            }
        }
        public void MoveToTop(int fromIndex)
        {
            BasemapLayer layer = _Layers[fromIndex];
            _Layers.RemoveAt(fromIndex);
            _Layers.Insert(0, layer);
        }
        public void MoveToBottom(int fromIndex)
        {
            BasemapLayer layer = _Layers[fromIndex];
            _Layers.RemoveAt(fromIndex);
            _Layers.Add(layer);
        }
        public void MoveUp(int fromIndex)
        {
            if ( (fromIndex < 0) || (fromIndex > _Layers.Count - 1) )
                throw new Exception("Layer index is out of range.");
            if (fromIndex == 0) return;
            int toIndex = fromIndex - 1;
            BasemapLayer layer = _Layers[fromIndex];
            _Layers.RemoveAt(fromIndex);
            _Layers.Insert(toIndex, layer);
        }
        public void MoveDown(int fromIndex)
        {
            if ((fromIndex < 0) || (fromIndex > _Layers.Count - 1))
                throw new Exception("Layer index is out of range.");
            int toIndex = fromIndex + 1;
            if (toIndex == _Layers.Count) return;
            BasemapLayer layer = _Layers[fromIndex];
            _Layers.RemoveAt(fromIndex);
            if (toIndex == _Layers.Count)
            { _Layers.Add(layer); }
            else
            { _Layers.Insert(toIndex, layer); }
        }
        public void RemoveLayer(string localName)
        {
            RemoveLayer(FindIndex(localName));
        }
        public void RemoveLayer(int index)
        {
            if (index < 0) return;
            _Layers.RemoveAt(index);
        }
        public void RemoveAllLayers()
        {
            _Layers.Clear();
        }
        public List<FeatureLayer> CreateBasemapLayers()
        {
            FeatureLayer layer = null;
            List<FeatureLayer> layers = new List<FeatureLayer>();
            for (int i = 0; i < this.LayerCount; i++)
            {
                layer = CreateBasemapLayerFromShapefile(this.BasemapDirectory, this[i]);
                layer.LayerName = this[i].Description;
                if (layer != null)
                { layers.Add(layer); }
            }

            return layers;

        }

        #endregion

        #region Protected Methods
        protected bool Load(string filename)
        {
            return true;
        }
        #endregion

        #region Private Methods
        private FeatureLayer CreateBasemapLayerFromShapefile(string directoryName, BasemapLayer basemapLayer)
        {

            FeatureLayer layer = null;
            string filename = Path.Combine(directoryName, basemapLayer.LocalName).Trim();
            if (string.IsNullOrEmpty(filename))
                throw new ArgumentException("The shapefile pathname is blank.");

            // Read ESRI shapefile and import features to a FeatureCollection
            FeatureCollection featureList = USGS.Puma.IO.EsriShapefileIO.Import(filename);

            if (featureList != null)
            {
                if (featureList.Count > 0)
                {
                    Feature f = featureList[0];
                    if (f.Geometry is IMultiLineString || f.Geometry is ILineString)
                    {
                        layer = new FeatureLayer(LayerGeometryType.Line);
                        ILineSymbol symbol = ((ISingleSymbolRenderer)(layer.Renderer)).Symbol as ILineSymbol;
                        symbol.Color = basemapLayer.Color;
                        symbol.Width = Convert.ToSingle(basemapLayer.Size);
                    }
                    else if (f.Geometry is IPolygon)
                    {
                        layer = new FeatureLayer(LayerGeometryType.Polygon);
                        ISolidFillSymbol symbol = ((ISingleSymbolRenderer)(layer.Renderer)).Symbol as ISolidFillSymbol;
                        symbol.Color = basemapLayer.Color;
                    }
                    else if (f.Geometry is IPoint)
                    {
                        layer = new FeatureLayer(LayerGeometryType.Point);
                        SimplePointSymbol symbol = (((layer.Renderer as SingleSymbolRenderer).Symbol) as SimplePointSymbol);
                        symbol.Color = basemapLayer.Color;
                        symbol.Size = Convert.ToSingle(basemapLayer.Size);
                        if (basemapLayer.Style == 1)
                        {
                            symbol.SymbolType = PointSymbolTypes.Circle;
                        }
                        else if (basemapLayer.Style == 2)
                        {
                            symbol.SymbolType = PointSymbolTypes.Square;
                        }
                        else
                        {
                            symbol.SymbolType = PointSymbolTypes.Circle;
                        }
                    }
                    else
                    {
                        throw new ArgumentException("Cannot create layer for the specified feature type.");
                    }

                    for (int i = 0; i < featureList.Count; i++)
                    {
                        layer.AddFeature(featureList[i]);
                    }

                    layer.Visible = basemapLayer.IsVisible;

                }
            }

            return layer;

        }

        #endregion



    }
}
