using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public enum LayerGeometryType
    {
        Mixed = 0,
        Point = 1,
        Line = 2,
        Polygon = 3,
    }
    public interface IFeatureLayer : IGraphicLayer
    {
        LayerGeometryType GeometryType { get; }
        IFeatureRenderer Renderer { get; set; }
        int FeatureCount {get;}
        void AddFeature(GeoAPI.Geometries.IGeometry geometry,USGS.Puma.NTS.Features.IAttributesTable attributes);
        void AddFeature(USGS.Puma.NTS.Features.Feature feature);
        void RemoveFeature(int index);
        void RemoveAll();
        void MoveUp(int fromIndex);
        void MoveDown(int fromIndex);
        void MoveToTop(int fromIndex);
        void ModeToBottom(int fromIndex);
        USGS.Puma.NTS.Features.Feature GetFeature(int index);
        USGS.Puma.NTS.Features.FeatureCollection GetFeatures();
        void UpdateFeature(USGS.Puma.NTS.Features.Feature feature, int index);
        void Update();
    }
}
