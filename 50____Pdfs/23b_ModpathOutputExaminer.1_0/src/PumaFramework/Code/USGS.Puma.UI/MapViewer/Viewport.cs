
using System;
using System.Collections.Generic;
using System.Text;
using System.Drawing;
using System.Drawing.Drawing2D;
using USGS.Puma.NTS;
using USGS.Puma.NTS.Geometries;
using GeoAPI.Geometries;

namespace USGS.Puma.UI.MapViewer
{
    public class Viewport
    {
        #region Private Fields
        #endregion

        #region Constructors
        public Viewport(Size viewportSize)
            : this(viewportSize, new Envelope(0, 1, 0, 1))
        {

        }
        public Viewport(Size viewportSize, IEnvelope targetWorldExtent)
        {
            Initialize(viewportSize, targetWorldExtent);
        }
        #endregion

        #region Public Properties

        private Size _ViewportSize;
        public Size ViewportSize
        {
            get { return _ViewportSize; }
            set
            {
                _ViewportSize = value;
                SetTargetWorldExtent(_TargetWorldExtent);
            }
        }

        private IEnvelope _TargetWorldExtent;
        public IEnvelope TargetWorldExtent
        {
            get { return _TargetWorldExtent; }
        }

        public IEnvelope WorldExtent
        {
            get { return CalculateViewportExtent(); }
        }

        //public double PixelSize
        //{
        //    get { return _ViewportExtent.Width/Convert.ToDouble(_DeviceSize.Width); }
        //}

        #endregion

        #region General Public Methods
        public void Initialize(Size viewportSize, IEnvelope targetWorldExtent)
        {
            _ViewportSize = viewportSize;
            SetTargetWorldExtent(targetWorldExtent.MinX,targetWorldExtent.MaxX,targetWorldExtent.MinY,targetWorldExtent.MaxY);
        }

        public void SetTargetWorldExtent(IEnvelope targetWorldExtent)
        {
            SetTargetWorldExtent(targetWorldExtent.MinX, targetWorldExtent.MaxX, targetWorldExtent.MinY, targetWorldExtent.MaxY);
        }
        public void SetTargetWorldExtent(ICoordinate center, double width, double height)
        {
            SetTargetWorldExtent(center.X - width / 2.0, center.X + width / 2.0, center.Y - height / 2.0, center.Y + height / 2.0);
        }
        public void SetTargetWorldExtent(ICoordinate center, double length)
        {
            length = length / 2.0;
            SetTargetWorldExtent(center.X - length, center.X + length, center.Y - length, center.Y + length);
        }
        public void SetTargetWorldExtent(double minX, double maxX, double minY, double maxY)
        {
            if (ViewportSize.Width == 0 || ViewportSize.Height == 0)
                throw new ArgumentException("Specified device size has 0 value for one or more dimensions.");

            // Set the target extent.
            _TargetWorldExtent = new Envelope(minX, maxX, minY, maxY);

        }


        public void CenterAt(ICoordinate center)
        {
            double minX = center.X - (WorldExtent.Width / 2.0);
            double maxX = center.X + (WorldExtent.Width / 2.0);
            double minY = center.Y - (WorldExtent.Height / 2.0);
            double maxY = center.Y + (WorldExtent.Height / 2.0);
            SetTargetWorldExtent(minX, maxX, minY, maxY);
        }
        public void Zoom(double zoomFactor)
        {
            if (zoomFactor == 1) return;
            if (zoomFactor < 0.1 || zoomFactor > 10) return;

            double dx;
            double dy;
            IEnvelope extent = new Envelope(_TargetWorldExtent);
            dx = extent.Width * (1.0 - zoomFactor) / zoomFactor / 2.0;
            dy = extent.Height * (1.0 - zoomFactor) / zoomFactor / 2.0;
            extent.ExpandBy(dx, dy);
            SetTargetWorldExtent(extent);
        }
        public Viewport GetCopy()
        {
            Viewport converter = new Viewport(this.ViewportSize, this.TargetWorldExtent);
            return converter;
        }
        #endregion

        #region Public Coordinate Conversion Methods
        public System.Drawing.PointF ToDevicePoint(ICoordinate coord)
        {
            return ToDevicePoint(coord.X, coord.Y);
        }
        public System.Drawing.PointF ToDevicePoint(double x, double y)
        {
            PointF pt = new PointF();
            double vpWidth = Convert.ToDouble(_ViewportSize.Width - 1);
            double vpHeight = Convert.ToDouble(_ViewportSize.Height - 1);
            pt.X = Convert.ToSingle(vpWidth * (x - WorldExtent.MinX) / (WorldExtent.MaxX-WorldExtent.MinX));
            pt.Y = Convert.ToSingle(vpHeight * (WorldExtent.MaxY - y) / (WorldExtent.MaxY-WorldExtent.MinY));
            return pt;
        }

        public System.Drawing.PointF[] ToDevicePoints(ILineString line)
        {
            if (line == null)
                throw new ArgumentNullException();

            return ToDevicePoints(line.Coordinates);
        }
        public System.Drawing.PointF[] ToDevicePoints(ICoordinate[] coords)
        {
            if (coords == null)
                throw new ArgumentNullException();
            PointF[] pts = new PointF[coords.Length];
            for (int i = 0; i < coords.Length; i++)
            {
                pts[i] = ToDevicePoint(coords[i]);
            }
            return pts;
        }

        public ICoordinate ToMapPoint(System.Drawing.Point devicePoint)
        {
            return ToMapPoint((float)devicePoint.X, (float)devicePoint.Y);
        }
        public ICoordinate ToMapPoint(PointF devicePoint)
        {
            return ToMapPoint(devicePoint.X, devicePoint.Y);
        }
        public ICoordinate ToMapPoint(int x, int y)
        {
            return ToMapPoint((float)x, (float)y);
        }
        public ICoordinate ToMapPoint(float x, float y)
        {
            double xx = Convert.ToDouble(x);
            double yy = Convert.ToDouble(y);
            double vpWidth = Convert.ToDouble(_ViewportSize.Width - 1);
            double vpHeight = Convert.ToDouble(_ViewportSize.Height - 1);

            ICoordinate pt = new Coordinate();
            pt.X = WorldExtent.MinX + (WorldExtent.MaxX - WorldExtent.MinX) * xx / vpWidth;
            pt.Y = WorldExtent.MaxY - (WorldExtent.MaxY - WorldExtent.MinY) * yy / vpHeight;
            return pt;
        }

        public ICoordinate[] ToMapPoints(System.Drawing.PointF[] points)
        {
            if (points == null)
                throw new ArgumentNullException();

            ICoordinate[] coords = new Coordinate[points.Length];
            for (int i = 0; i < points.Length; i++)
            {
                coords[i] = ToMapPoint(points[i]);
            }
            return coords;
        }
        public ICoordinate[] ToMapPoints(System.Drawing.Point[] points)
        {
            if (points == null)
                throw new ArgumentNullException();

            ICoordinate[] coords = new Coordinate[points.Length];
            for (int i = 0; i < points.Length; i++)
            {
                coords[i] = ToMapPoint(points[i].X, points[i].Y);
            }
            return coords;


        }

        public RectangleF ToDeviceRectangle(double left, double right, double bottom, double top)
        {
            double vpWidth = Convert.ToDouble(_ViewportSize.Width - 1);
            double vpHeight = Convert.ToDouble(_ViewportSize.Height - 1);

            PointF pt = ToDevicePoint(left, top);
            float w = Convert.ToSingle(vpWidth *(right - left) / (WorldExtent.MaxX-WorldExtent.MinX));
            float h = Convert.ToSingle(vpHeight * (top - bottom) / (WorldExtent.MaxY-WorldExtent.MinY));
            return new RectangleF(pt.X, pt.Y, w, h);
        }
        public RectangleF ToDeviceRectangle(IEnvelope extent)
        {
            return ToDeviceRectangle(extent.MinX, extent.MaxX, extent.MinY, extent.MaxY);
        }

        #endregion

        #region Public Drawing Methods
        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="g"></param>
        /// <param name="pen"></param>
        /// <param name="brush"></param>
        /// <param name="inflate"></param>
        public void DrawEnvelope(IEnvelope shape, System.Drawing.Graphics g, Pen pen, Brush brush, bool inflate)
        {
            RectangleF rect = ToDeviceRectangle(shape);
            if (inflate) rect.Inflate(0.5f, 0.5f);
            if (brush != null)
                g.FillRectangle(brush, rect);
            if (pen != null)
                g.DrawRectangle(pen, rect.X, rect.Y, rect.Width, rect.Height);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="line"></param>
        /// <param name="g"></param>
        /// <param name="pen"></param>
        public void DrawLineString(ILineString line, System.Drawing.Graphics g, Pen pen)
        {
            PointF[] pts = ToDevicePoints(line);
            if (pen != null)
                g.DrawLines(pen, pts);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="coords"></param>
        /// <param name="g"></param>
        /// <param name="pen"></param>
        public void DrawLineString(ICoordinate[] coords, System.Drawing.Graphics g, Pen pen)
        {
            PointF[] pts = ToDevicePoints(coords);
            if (pen != null)
                g.DrawLines(pen, pts);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="pts"></param>
        /// <param name="g"></param>
        /// <param name="pen"></param>
        public void DrawLineString(PointF[] pts, System.Drawing.Graphics g, Pen pen)
        {
            if (pen != null)
                g.DrawLines(pen, pts);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="polygon"></param>
        /// <param name="g"></param>
        /// <param name="pen"></param>
        /// <param name="brush"></param>
        public void DrawPolygon(IPolygon polygon, System.Drawing.Graphics g, Pen pen, Brush brush)
        {
            if (polygon == null)
                throw new ArgumentNullException("The specified polygon is null.");
            if (!polygon.IsSimple)
                throw new ArgumentException("Non-simple polygons are not supported.");

            PointF[] pts = ToDevicePoints(polygon.Shell.Coordinates);
            if (brush != null)
                g.FillPolygon(brush, pts);
            if (pen != null)
                g.DrawPolygon(pen, pts);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="pts"></param>
        /// <param name="g"></param>
        /// <param name="pen"></param>
        /// <param name="brush"></param>
        public void DrawPolygon(PointF[] pts, System.Drawing.Graphics g, Pen pen, Brush brush)
        {
            if (brush != null)
                g.FillPolygon(brush, pts);
            if (pen != null)
                g.DrawPolygon(pen, pts);

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="multiLine"></param>
        /// <param name="g"></param>
        /// <param name="pen"></param>
        public void DrawMultiLineString(IMultiLineString multiLine, System.Drawing.Graphics g, System.Drawing.Pen pen)
        {
            if (multiLine == null)
                throw new ArgumentNullException("The specified geometry object is null.");
            if (g == null)
                throw new ArgumentNullException("The specified graphics canvas is null.");
            if (pen == null)
                throw new ArgumentNullException("The specified drawing pen is null.");

            for (int i = 0; i < multiLine.Count; i++)
            {
                DrawLineString((ILineString)multiLine[i], g, pen);
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="point"></param>
        /// <param name="g"></param>
        /// <param name="pen"></param>
        /// <param name="brush"></param>
        /// <param name="size"></param>
        /// <param name="symbolType"></param>
        public void DrawCirclePoint(IPoint point, System.Drawing.Graphics g, System.Drawing.Pen pen, System.Drawing.Brush brush, float size)
        {
            PointF pt = ToDevicePoint(point.Coordinate);
            float x = pt.X - size;
            float y = pt.Y - size;
            float width = (2.0f) * size;
            float height = width;
            if (brush != null)
            {
                g.FillEllipse(brush, x, y, width, height);
            }
            if (pen != null)
            {
                g.DrawEllipse(pen, x, y, width, height);
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="pt"></param>
        /// <param name="g"></param>
        /// <param name="pen"></param>
        /// <param name="brush"></param>
        /// <param name="size"></param>
        public void DrawCirclePoint(PointF pt, System.Drawing.Graphics g, System.Drawing.Pen pen, System.Drawing.Brush brush, float size)
        {
            float x = pt.X - size;
            float y = pt.Y - size;
            float width = (2.0f) * size;
            float height = width;
            if (brush != null)
            {
                g.FillEllipse(brush, x, y, width, height);
            }
            if (pen != null)
            {
                g.DrawEllipse(pen, x, y, width, height);
            }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="point"></param>
        /// <param name="g"></param>
        /// <param name="pen"></param>
        /// <param name="brush"></param>
        /// <param name="size"></param>
        public void DrawSquarePoint(IPoint point, System.Drawing.Graphics g, System.Drawing.Pen pen, System.Drawing.Brush brush, float size)
        {
            PointF pt = ToDevicePoint(point.Coordinate);
            float x = pt.X - size;
            float y = pt.Y - size;
            float width = (2.0f) * size;
            float height = width;
            if (brush != null)
            {
                g.FillRectangle(brush, x, y, width, height);
            }
            if (pen != null)
            {
                g.DrawRectangle(pen, x, y, width, height);
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="pt"></param>
        /// <param name="g"></param>
        /// <param name="pen"></param>
        /// <param name="brush"></param>
        /// <param name="size"></param>
        public void DrawSquarePoint(PointF pt, System.Drawing.Graphics g, System.Drawing.Pen pen, System.Drawing.Brush brush, float size)
        {
            float x = pt.X - size;
            float y = pt.Y - size;
            float width = (2.0f) * size;
            float height = width;
            if (brush != null)
            {
                g.FillRectangle(brush, x, y, width, height);
            }
            if (pen != null)
            {
                g.DrawRectangle(pen, x, y, width, height);
            }

        }
        #endregion

        #region Private Methods
        private IEnvelope CalculateViewportExtent()
        {
            if (ViewportSize.Width == 0 || ViewportSize.Height == 0)
                throw new ArgumentException("Specified device size has 0 value for one or more dimensions.");

            // Define some variables
            double pixelSize;
            double x1 = _TargetWorldExtent.MinX;
            double x2 = _TargetWorldExtent.MaxX;
            double y1 = _TargetWorldExtent.MinY;
            double y2 = _TargetWorldExtent.MaxY;
            double center;
            double vpWidth = Convert.ToDouble(_ViewportSize.Width - 1);
            double vpHeight = Convert.ToDouble(_ViewportSize.Height - 1);
            double worldAspect = (x2 - x1) / (y2 - y1);
            double vpAspect = vpWidth / vpHeight;

            // Use the target extent to set the viewport extent.
            if (worldAspect > vpAspect)
            {
                pixelSize = (x2 - x1) / vpWidth;
                center = (y1 + y2) / 2.0;
                y1 = center - (vpHeight * pixelSize) / 2.0;
                y2 = center + (vpHeight * pixelSize) / 2.0;
            }
            else
            {
                pixelSize = (y2 - y1) / vpHeight;
                center = (x1 + x2) / 2.0;
                x1 = center - (vpWidth * pixelSize) / 2.0;
                x2 = center + (vpWidth * pixelSize) / 2.0;
            }

            return new Envelope(x1, x2, y1, y2);

        }

        #endregion

    }
}
