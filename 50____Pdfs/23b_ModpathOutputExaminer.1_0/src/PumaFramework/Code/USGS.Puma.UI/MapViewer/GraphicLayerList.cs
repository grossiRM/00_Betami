using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.NTS.Geometries;
using GeoAPI.Geometries;

namespace USGS.Puma.UI.MapViewer
{
    public class GraphicLayerList : IGraphicLayerList
    {
        private List<GraphicLayer> _GraphicsLayers;

        //public event EventHandler<LayerEventArgs> LayerVisibilityChanged;
        //protected virtual void OnLayerVisibilityChanged(GraphicLayer layer)
        //{
        //    if (_GraphicsLayers.Contains(layer))
        //    {
        //        int index = _GraphicsLayers.IndexOf(layer);
        //        this.LayerVisibilityChanged(this, new LayerEventArgs(index));
        //    }
        //}

        public GraphicLayerList()
        {
            _GraphicsLayers = new List<GraphicLayer>();
        }
        

        #region IGraphicsLayers Members
        public int LayerCount
        { get { return _GraphicsLayers.Count; } }

        public void ClearLayers()
        {
            //foreach (GraphicLayer layer in _GraphicsLayers)
            //{
            //    if (layer != null)
            //    { layer.LayerVisibilityChanged -= layer_LayerVisibilityChanged; }
            //}
            _GraphicsLayers.Clear();
        }
        public void AddLayer(GraphicLayer layer)
        {
            _GraphicsLayers.Insert(0, layer);
            //layer.LayerVisibilityChanged += new EventHandler(layer_LayerVisibilityChanged);
        }
        public void RemoveLayer(int index)
        {
            //if (_GraphicsLayers[index] != null)
            //{ _GraphicsLayers[index].LayerVisibilityChanged -= layer_LayerVisibilityChanged; }
            _GraphicsLayers.RemoveAt(index);
        }
        public void MoveToTop(int fromIndex)
        {
            throw new NotImplementedException();
        }
        public void MoveToBottom(int fromIndex)
        {
            throw new NotImplementedException();
        }
        public void MoveUp(int fromIndex)
        {
            throw new NotImplementedException();
        }
        public void MoveDown(int fromIndex)
        {
            throw new NotImplementedException();
        }
        public GraphicLayer GetLayer(int index)
        {
            return _GraphicsLayers[index];
        }

        public GeoAPI.Geometries.IEnvelope FullExtent
        {
            get 
            {
                IEnvelope extent = new Envelope();
                foreach (GraphicLayer layer in _GraphicsLayers)
                {
                    extent.ExpandToInclude(layer.Extent);
                }
                return extent;
            }
        }

        #endregion

        //private void layer_LayerVisibilityChanged(object sender, EventArgs e)
        //{
        //    OnLayerVisibilityChanged(sender as GraphicLayer);
        //}

    }

    public class LayerEventArgs : EventArgs
    {
        public LayerEventArgs(int index)
            : base()
        {
            _LayerIndex = index;
        }

        private int _LayerIndex;

        public int LayerIndex
        {
            get { return _LayerIndex; }
        }

    }
}
