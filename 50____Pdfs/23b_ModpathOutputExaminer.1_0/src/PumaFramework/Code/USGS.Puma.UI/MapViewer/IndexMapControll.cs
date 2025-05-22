using System;
using System.Collections.Generic;
using System.Text;
using GeoAPI.Geometries;
using USGS.Puma.NTS.Geometries;

namespace USGS.Puma.UI.MapViewer
{
    public class IndexMapControl : System.Windows.Forms.Panel
    {
        private System.Drawing.Image _Image = null;
        private System.Drawing.RectangleF _ImageRect;
        private Viewport _VP = null;
        private System.Drawing.Size _SizeLastPaint;
        private IEnvelope _SourceFullExtentLastPaint = null;

        public IndexMapControl()
            : base()
        {
            this.DoubleBuffered = true;
            SuppressMapImageUpdate = false;
            _VP = new Viewport(this.ClientSize);
            base.MouseClick += new System.Windows.Forms.MouseEventHandler(IndexMapControl_MouseClick);
            
            //// Retrieve and load the ReCenter.cur cursor that is stored as a project resource.
            //System.IO.MemoryStream stream = new System.IO.MemoryStream(USGS.Puma.UI.Properties.Resources.ReCenterCur);
            //this.Cursor = new System.Windows.Forms.Cursor(stream);
        }

        void IndexMapControl_MouseClick(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            if (_SourceMap != null)
            {
                ICoordinate pt = _VP.ToMapPoint(e.X, e.Y);
                _SourceMap.Center = pt;
            }
        }

        private MapControl _SourceMap;
        public MapControl SourceMap
        {
            get { return _SourceMap; }
            set
            {
                if (value == null)
                {
                    if (_SourceMap != null)
                    { 
                        _SourceMap.RefreshCompleted -= _SourceMap_RefreshCompleted;
                    }
                    _SourceMap = null;
                    if (_Image != null)
                        _Image.Dispose();
                    _Image = null;
                    _ImageRect = new System.Drawing.RectangleF();
                    this.Refresh();
                }
                else
                {
                    if (_SourceMap != null)
                    { 
                        _SourceMap.RefreshCompleted -= _SourceMap_RefreshCompleted;
                    }
                    _SourceMap = value;
                    _SourceMap.RefreshCompleted += new EventHandler<MapControlRefreshCompletedArgs>(_SourceMap_RefreshCompleted);
                    UpdateMapImage();
                }
            }
        }

        private void _SourceMap_RefreshCompleted(object sender, MapControlRefreshCompletedArgs e)
        {
            if (_SourceMap != null)
            {
                if (!SuppressMapImageUpdate)
                {
                    if (_SourceFullExtentLastPaint == null)
                    { UpdateMapImage(); }
                    else if (_SourceMap.ViewportSize != _SizeLastPaint)
                    { UpdateMapImage(); }
                    else if (_SourceMap.FullExtent.MinX != _SourceFullExtentLastPaint.MinX)
                    { UpdateMapImage(); }
                    else if (_SourceMap.FullExtent.MaxX != _SourceFullExtentLastPaint.MaxX)
                    { UpdateMapImage(); }
                    else if (_SourceMap.FullExtent.MinY != _SourceFullExtentLastPaint.MinY)
                    { UpdateMapImage(); }
                    else if (_SourceMap.FullExtent.MaxY != _SourceFullExtentLastPaint.MaxY)
                    { UpdateMapImage(); }
                    else
                    { this.Refresh(); }
                }
            }
        }

        private bool _SuppressMapImageUpdate;
        public bool SuppressMapImageUpdate
        {
            get { return _SuppressMapImageUpdate; }
            set { _SuppressMapImageUpdate = value; }
        }

        private void UpdateMapImageInternal()
        {
            if (_Image != null)
            { _Image.Dispose(); }
            if (_SourceMap != null)
            {
                _VP.Initialize(this.ClientSize, GetViewportExtent());
                System.Drawing.RectangleF rect = GetImageRect();
                System.Drawing.Size size = new System.Drawing.Size(Convert.ToInt32(rect.Width), Convert.ToInt32(rect.Height));
                _Image = _SourceMap.RenderAsImage(size, _SourceMap.FullExtent);
                _ImageRect = rect;
            }

            if (_SourceMap != null)
            {
                _SizeLastPaint = _SourceMap.ViewportSize;
                _SourceFullExtentLastPaint = new Envelope(_SourceMap.FullExtent);
            }

        }

        public void UpdateMapImage()
        {
            UpdateMapImageInternal();
            this.Refresh();

            //if (_Image != null)
            //{ _Image.Dispose(); }
            //if (_SourceMap != null)
            //{
            //    _VP.Initialize(this.ClientSize, GetViewportExtent());
            //    System.Drawing.RectangleF rect = GetImageRect();
            //    System.Drawing.Size size = new System.Drawing.Size(Convert.ToInt32(rect.Width), Convert.ToInt32(rect.Height));
            //    _Image = _SourceMap.RenderAsImage(size, _SourceMap.FullExtent);
            //    _ImageRect = rect;
            //}
            //this.Refresh();

            //if (_SourceMap != null)
            //{
            //    _SizeLastPaint = _SourceMap.ViewportSize;
            //    _SourceFullExtentLastPaint = new Envelope(_SourceMap.FullExtent);
            //}

        }

        protected System.Drawing.RectangleF GetImageRect()
        {
            if (_SourceMap == null)
                return new System.Drawing.RectangleF();

            float sourceVpAspect = Convert.ToSingle(_SourceMap.ViewportSize.Width) / Convert.ToSingle(_SourceMap.ViewportSize.Height);
            float vpWidth = Convert.ToSingle(_VP.ViewportSize.Width);
            float vpHeight = Convert.ToSingle(_VP.ViewportSize.Height);
            float vpAspect = vpWidth / vpHeight;

            if (sourceVpAspect > vpAspect)
            {
                float width = vpWidth;
                float height = vpWidth / sourceVpAspect;
                float x = 0f;
                float y = (vpHeight - height) / 2.0f;
                return new System.Drawing.RectangleF(x, y, width, height);
            }
            else
            {
                float height = vpHeight;
                float width = vpHeight * sourceVpAspect;
                float y = 0f;
                float x = (vpWidth - width) / 2.0f;
                return new System.Drawing.RectangleF(x, y, width, height);
            }
            
        }

        protected IEnvelope GetViewportExtent()
        {
            if (_SourceMap == null)
                return null;

            if (_ImageRect.X == 0.0f)
            {
                IEnvelope sourceVpExtent = _SourceMap.FullExtent;
                double minX = sourceVpExtent.MinX;
                double maxX = sourceVpExtent.MaxX;
                double delta = Convert.ToDouble(_ImageRect.Y / _ImageRect.Height) * sourceVpExtent.Height;
                double minY = sourceVpExtent.MinY - delta;
                double maxY = sourceVpExtent.MaxY + delta;
                return new Envelope(minX, maxX, minY, maxY);
            }
            else
            {
                IEnvelope sourceVpExtent = _SourceMap.FullExtent;
                double minY = sourceVpExtent.MinY;
                double maxY = sourceVpExtent.MaxY;
                double delta = Convert.ToDouble(_ImageRect.X / _ImageRect.Width) * sourceVpExtent.Width;
                double minX = sourceVpExtent.MinX - delta;
                double maxX = sourceVpExtent.MaxX + delta;
                return new Envelope(minX, maxX, minY, maxY);
            }
        }
        
        protected override void OnPaint(System.Windows.Forms.PaintEventArgs e)
        {
            if ((this.ClientSize.Width != _SizeLastPaint.Width) || (this.ClientSize.Height != _SizeLastPaint.Height))
            {
                UpdateMapImageInternal();
            }

            // Draw map image
            if (_Image != null)
            {
                e.Graphics.DrawImage(_Image, _ImageRect);
            }

            // Draw map extent
            if (_SourceMap != null)
            {
                if (_SourceMap.LayerCount > 0)
                {
                    System.Drawing.Pen pen = new System.Drawing.Pen(System.Drawing.Color.DarkGray, 2.0f);
                    _VP.SetTargetWorldExtent(GetViewportExtent());
                    _VP.DrawEnvelope(_SourceMap.ViewportExtent, e.Graphics, pen, null, false);
                    if (pen != null)
                        pen.Dispose();
                }
            }

            // Execute Panel OnPaint method
            base.OnPaint(e);

        }

        #region EventHandlers
        private void _SourceMap_MapExtentChanged(object sender, EventArgs e)
        {
            this.Refresh();
        }
        #endregion

        private void InitializeComponent()
        {
            this.SuspendLayout();
            // 
            // IndexMapControl
            // 
            this.BackColor = System.Drawing.SystemColors.ControlDark;
            this.ResumeLayout(false);

        }

    }
}
