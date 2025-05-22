using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public interface IGraphicLayerList
    {
        void ClearLayers();
        void AddLayer(GraphicLayer layer);
        void RemoveLayer(int index);
        void MoveToTop(int fromIndex);
        void MoveToBottom(int fromIndex);
        void MoveUp(int fromIndex);
        void MoveDown(int fromIndex);
        GraphicLayer GetLayer(int index);
        int LayerCount { get; }
        GeoAPI.Geometries.IEnvelope FullExtent { get; }
    }
}
