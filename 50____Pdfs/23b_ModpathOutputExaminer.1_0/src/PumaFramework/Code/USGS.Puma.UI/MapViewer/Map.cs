using System;
using System.Collections.Generic;
using System.Text;
using System.Drawing;
using GeoAPI.Geometries;
using USGS.Puma.NTS.Geometries;

namespace USGS.Puma.UI.MapViewer
{
    public class Map : GraphicLayerList, IMap, IDisposable
    {
        #region Private Fields
        private Viewport _Viewport = null;
        private RendererHelper _RH = null;
        //private GraphicLayerList _GraphicsLayers = null;
        #endregion

        #region Events

        #endregion

        #region Constructors
        public Map() 
        {
            _RH = new RendererHelper();
            _MapBackgroundColor = System.Drawing.Color.White;
            _Viewport = new Viewport(new System.Drawing.Size(100, 100), new Envelope(0, 1, 0, 1));
        }

        public Map(System.Drawing.Size size)
            : this()
        {
            this.ViewportSize = size;
        }
        public Map(IMap map) 
            : this()
        {
            if (map == null)
            {
                for (int i = map.LayerCount - 1; i > map.LayerCount; i--)
                {
                    this.AddLayer(map.GetLayer(i));
                }
                this.MapBackgroundColor = map.MapBackgroundColor;
                this.SetViewport(map.ViewportSize, map.ViewportExtent);
            }
        }
        public Map(IMap map, System.Drawing.Size size)
            : this(map)
        {
            this.ViewportSize = size;
        }
        public Map(IMap map, System.Drawing.Size size,GeoAPI.Geometries.IEnvelope extent) 
            : this(map)
        {
            if (map != null)
            { this.SetViewport(size, extent); }
        }

        #endregion

        #region IMap Members
        private System.Drawing.Color _MapBackgroundColor;
        public System.Drawing.Color MapBackgroundColor
        {
            get
            {
                return _MapBackgroundColor;
            }
            set
            {
                _MapBackgroundColor = value;
            }
        }
        public System.Drawing.Size ViewportSize
        {
            get
            {
                return _Viewport.ViewportSize;
            }
            set
            {
                _Viewport.ViewportSize = value;
            }
        }
        public GeoAPI.Geometries.IEnvelope MapExtent
        {
            get
            {
                return _Viewport.TargetWorldExtent;
            }
            set
            {
                _Viewport.SetTargetWorldExtent(value);
            }
        }
        public GeoAPI.Geometries.IEnvelope ViewportExtent
        {
            get { return _Viewport.WorldExtent; }
        }
        public GeoAPI.Geometries.ICoordinate Center
        {
            get { return _Viewport.WorldExtent.Centre; }
            set
            {
                if (value != null)
                {
                    _Viewport.CenterAt(value);
                    // fire ViewportChanged event
                }
            }
        }
        public void SetViewport(System.Drawing.Size size, GeoAPI.Geometries.IEnvelope targetExtent)
        {
            _Viewport.Initialize(size, targetExtent);
            // fire ViewportChanged event
        }
        public void SetExtent(ICoordinate center, double width, double height)
        {
            _Viewport.SetTargetWorldExtent(center, width, height);
        }
        public void SetExtent(ICoordinate center, double length)
        {
            _Viewport.SetTargetWorldExtent(center, length);
        }
        public void SetExtent(double minX, double maxX, double minY, double maxY)
        {
            _Viewport.SetTargetWorldExtent(minX, maxX, minY, maxY);
        }
        public void SizeToFullExtent()
        {
            if (FullExtent == null)
            { MapExtent = new Envelope(0, 1, 0, 1); }
            else
            { MapExtent = new Envelope(FullExtent); }

        }
        public GeoAPI.Geometries.ICoordinate ToMapPoint(int x, int y)
        {
            return _Viewport.ToMapPoint(x, y);
        }
        public void Zoom(double factor)
        {
            _Viewport.Zoom(factor);
        }
        public void Zoom(double factor, double x, double y)
        {
            _Viewport.Zoom(factor);
            _Viewport.CenterAt(new Coordinate(x, y));
        }
        public System.Drawing.Image RenderAsImage()
        {
            System.Drawing.Image img = null;
            System.Drawing.Graphics g = null;
            try
            {
                img = new System.Drawing.Bitmap(this.ViewportSize.Width, this.ViewportSize.Height);
                g = System.Drawing.Graphics.FromImage(img);
                g.Clear(this.MapBackgroundColor);
                g.PageUnit = System.Drawing.GraphicsUnit.Pixel;

                // Render layers
                if (this.LayerCount > 0)
                {
                    _RH.Render(this as IGraphicLayerList, g, _Viewport);
                }

                return img;

            }
            catch (Exception)
            {
                throw;
            }
            finally
            {
                if (g != null)
                { g.Dispose(); }
            }

        }
        public System.Drawing.Image RenderAsImage(GeoAPI.Geometries.IEnvelope extent)
        {
            throw new NotImplementedException();
        }
        public System.Drawing.Image RenderAsImage(System.Drawing.Size size)
        {
            Viewport vp = new Viewport(size, this.MapExtent);
            System.Drawing.Image img = null;
            System.Drawing.Graphics g = null;
            try
            {
                img = new System.Drawing.Bitmap(vp.ViewportSize.Width, vp.ViewportSize.Height);
                g = System.Drawing.Graphics.FromImage(img);
                g.Clear(this.MapBackgroundColor);
                g.PageUnit = System.Drawing.GraphicsUnit.Pixel;

                // Render layers
                if (this.LayerCount > 0)
                {
                    _RH.Render(this as IGraphicLayerList, g, vp);
                }

                return img;

            }
            catch (Exception)
            {
                throw;
            }
            finally
            {
                if (g != null)
                { g.Dispose(); }
            }

        }
        public System.Drawing.Image RenderAsImage(System.Drawing.Size size, GeoAPI.Geometries.IEnvelope extent)
        {
            System.Drawing.Image img = null;
            System.Drawing.Graphics g = null;
            try
            {
                Viewport vp = new Viewport(size, extent);
                img = new System.Drawing.Bitmap(vp.ViewportSize.Width, vp.ViewportSize.Height);
                g = System.Drawing.Graphics.FromImage(img);
                g.Clear(this.MapBackgroundColor);
                g.PageUnit = System.Drawing.GraphicsUnit.Pixel;

                // Render layers
                if (this.LayerCount > 0)
                {
                    _RH.Render(this as IGraphicLayerList, g, vp);
                }

                return img;

            }
            catch (Exception)
            {
                throw;
            }
            finally
            {
                if (g != null)
                { g.Dispose(); }
            }

        }
        #endregion

        public Viewport GetViewport()
        {
            return _Viewport;
        }

        #region IDisposable Members

        public void Dispose()
        {
        }

        #endregion
    }
}
