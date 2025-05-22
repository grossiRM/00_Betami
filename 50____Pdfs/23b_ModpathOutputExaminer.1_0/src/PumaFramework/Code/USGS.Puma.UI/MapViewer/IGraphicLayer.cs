using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public enum GraphicLayerType
    {
        VectorLayer = 0,
        ImageLayer = 1
    }
    public interface IGraphicLayer
    {
        GraphicLayerType LayerType { get; }
        double MinVisible { get; set; }
        double MaxVisible { get; set; }
        bool Visible { get; set; }
        string LayerName { get; set; }
        GeoAPI.Geometries.IEnvelope Extent { get; }
        int SRID { get; set; }
    }
}
