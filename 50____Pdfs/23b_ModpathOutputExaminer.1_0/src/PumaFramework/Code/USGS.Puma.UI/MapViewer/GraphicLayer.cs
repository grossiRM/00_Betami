using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public abstract class GraphicLayer : IGraphicLayer
    {
        internal GraphicLayer()
        {
            _MinVisible = 0.0;
            _MaxVisible = double.MaxValue;
            Visible = true;
            _LayerName = "";

        }

        //public event EventHandler LayerVisibilityChanged;

        #region IGraphicLayer Members

        public abstract GraphicLayerType LayerType
        {
            get;
        }

        private double _MinVisible;
        public double MinVisible
        {
            get
            {
                return _MinVisible;
            }
            set
            {
                if (value > MaxVisible)
                {
                    _MinVisible = MaxVisible;
                    _MaxVisible = value;
                }
                else
                { _MinVisible = value; }
            }
        }

        private double _MaxVisible;
        public double MaxVisible
        {
            get
            {
                return _MaxVisible;
            }
            set
            {
                if (value < MinVisible)
                {
                    _MaxVisible = _MinVisible;
                    _MinVisible = value;
                }
                else
                { _MaxVisible = value; }

            }
        }

        private bool _Visible;
        public bool Visible
        {
            get
            {
                return _Visible;
            }
            set
            {
                if (_Visible != value)
                {
                    _Visible = value;
                    //OnLayerVisibilityChanged(new EventArgs());
                }
            }
        }

        private string _LayerName;
        public string LayerName
        {
            get
            {
                return _LayerName; ;
            }
            set
            {
                _LayerName = value;
            }
        }

        public abstract GeoAPI.Geometries.IEnvelope Extent
        {
            get;
        }

        public abstract int SRID
        {
            get;
            set;
        }

        //protected void OnLayerVisibilityChanged(EventArgs e)
        //{
        //    EventHandler handler = LayerVisibilityChanged;
        //    if (handler != null)
        //    {
        //        LayerVisibilityChanged(this, e);
        //    }
        //}

        #endregion
    }

}
