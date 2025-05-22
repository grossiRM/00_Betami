using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public interface IMap : IGraphicLayerList
    {
        System.Drawing.Color MapBackgroundColor { get; set; }
        System.Drawing.Size ViewportSize { get; set; }
        GeoAPI.Geometries.IEnvelope MapExtent { get; set; }
        GeoAPI.Geometries.IEnvelope ViewportExtent { get; }
        GeoAPI.Geometries.ICoordinate Center { get; set; }
        void SetViewport(System.Drawing.Size size, GeoAPI.Geometries.IEnvelope targetExtent);
        void SetExtent(GeoAPI.Geometries.ICoordinate center, double width, double height);
        void SetExtent(GeoAPI.Geometries.ICoordinate center, double length);
        void SetExtent(double minX, double maxX, double minY, double maxY);
        void SizeToFullExtent();
        GeoAPI.Geometries.ICoordinate ToMapPoint(int x, int y);
        void Zoom(double factor);
        System.Drawing.Image RenderAsImage();
        System.Drawing.Image RenderAsImage(GeoAPI.Geometries.IEnvelope extent);
        System.Drawing.Image RenderAsImage(System.Drawing.Size size);
        System.Drawing.Image RenderAsImage(System.Drawing.Size size, GeoAPI.Geometries.IEnvelope extent);
    }
}
