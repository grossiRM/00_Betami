using System;
using System.Collections.Generic;
using System.Text;
using GeoAPI.Geometries;
using USGS.Puma.NTS.Geometries;
using USGS.Puma.NTS.Features;

namespace USGS.Puma.UI.MapViewer
{
    public class FeatureLayer : GraphicLayer, IFeatureLayer
    {
        protected IEnvelope _Extent = null;
        protected List<Feature> _Features = null;
        protected ColorRamp _RandomRamp = null;
        protected float _RandomPosition;

        #region Constructors
        public FeatureLayer(LayerGeometryType geometryType) : base()
        {
            _GeometryType = geometryType;
            _Features = new List<Feature>();
            _Extent = new Envelope();
            _RandomRamp = ColorRamp.Rainbow5;
            Random rand = new Random();
            _RandomPosition = Convert.ToSingle(rand.NextDouble());

            switch (GeometryType)
            {
                case LayerGeometryType.Mixed:
                    throw new NotImplementedException("Mixed geometry");
                    break;
                case LayerGeometryType.Point:
                    _Renderer = new SingleSymbolRenderer(new SimplePointSymbol(PointSymbolTypes.Circle, _RandomRamp.GetColor(_RandomPosition), 1.0f));
                    break;
                case LayerGeometryType.Line:
                    _Renderer = new SingleSymbolRenderer(new LineSymbol(_RandomRamp.GetColor(_RandomPosition), System.Drawing.Drawing2D.DashStyle.Solid, 1.0f));
                    break;
                case LayerGeometryType.Polygon:
                    _Renderer = new SingleSymbolRenderer(new SolidFillSymbol(_RandomRamp.GetColor(_RandomPosition)));
                    break;
                default:
                    break;
            }
        }
        public FeatureLayer(LayerGeometryType geometryType, Feature[] features)
            : this(geometryType)
        {
            foreach (Feature f in features)
            {
                AddFeature(f as Feature);
            }
        }
        #endregion

        public override IEnvelope Extent
        {
            get
            {
                if (_Extent == null)
                    Update();
                return _Extent;
            }
        }

        private LayerGeometryType _GeometryType;
        public LayerGeometryType GeometryType
        { get { return _GeometryType; } }

        protected IFeatureRenderer _Renderer;
        public IFeatureRenderer Renderer
        {
            get
            {
                return _Renderer;
            }
            set
            {
                if (value == null)
                { _Renderer = null; }
                else
                {
                    if ( IsValidSymbolType(value.SymbolType))
                    { _Renderer = value; }
                    else
                    { throw new ArgumentException("Specified renderer symbol type is incorrect or unsupported."); }
                }
            }
        }

        protected virtual bool IsValidSymbolType(SymbolType symbolType)
        {
            if (symbolType == null)
                return false;

            switch (GeometryType)
            {
                case LayerGeometryType.Mixed:
                    return false;
                    break;
                case LayerGeometryType.Point:
                    return (symbolType == SymbolType.PointSymbol);
                    break;
                case LayerGeometryType.Line:
                    return (symbolType == SymbolType.LineSymbol);
                    break;
                case LayerGeometryType.Polygon:
                    return (symbolType == SymbolType.FillSymbol);
                    break;
                default:
                    return false;
                    break;
            }
        }

        public int FeatureCount
        {
            get { return _Features.Count; }
        }

        public virtual void AddFeature(IGeometry geometry, IAttributesTable attributes)
        {
            if (geometry == null)
                throw new ArgumentNullException("The Geometry argument is null.");
            if (attributes == null)
                throw new ArgumentNullException("The attributes argument is null.");
            if (!IsValidGeometry(geometry))
                throw new ArgumentException("Geometry is not a LineString or MutliLineString.");
            AddFeature(new Feature(geometry, attributes));
        }

        public virtual void AddFeature(Feature feature)
        {
            if (feature == null)
                throw new ArgumentNullException("The feature argument is null.");
            if (feature.Geometry == null)
                throw new ArgumentNullException("The feature geometry property is null.");
            if (feature.Attributes == null)
                throw new ArgumentNullException("The feature attributes property argument is null.");
            if (!IsValidGeometry(feature.Geometry))
                throw new ArgumentException("Geometry is not a LineString or MutliLineString.");

            _Extent = null;
            _Features.Add(feature);
         
        }

        protected virtual bool IsValidGeometry(IGeometry geometry)
        {
            if (geometry == null)
                return false;

            switch (GeometryType)
            {
                case LayerGeometryType.Mixed:
                    return false;
                    break;
                case LayerGeometryType.Point:
                    return ((geometry is IPoint) || (geometry is IMultiPoint));
                    break;
                case LayerGeometryType.Line:
                    return ((geometry is ILineString) || (geometry is IMultiLineString));
                    break;
                case LayerGeometryType.Polygon:
                    return (geometry is IPolygon);
                    break;
                default:
                    return false;
                    break;
            }

        }

        public void RemoveFeature(int index)
        {
            _Extent = null;
            _Features.RemoveAt(index);
        }

        public void RemoveAll()
        {
            _Extent = null;
            _Features.Clear();
        }

        public void MoveUp(int fromIndex)
        {
            if (fromIndex < 0)
                throw new IndexOutOfRangeException();
            if (fromIndex > _Features.Count - 1)
                throw new IndexOutOfRangeException();
            if (fromIndex == 0)
                return;
            int i = fromIndex - 1;
            Feature toFeature = _Features[i];
            Feature fromFeature = _Features[fromIndex];
            _Features[i] = fromFeature;
            _Features[fromIndex] = toFeature;
        }

        public void MoveDown(int fromIndex)
        {
            if (fromIndex < 0)
                throw new IndexOutOfRangeException();
            if (fromIndex > _Features.Count - 1)
                throw new IndexOutOfRangeException();
            if (fromIndex == _Features.Count - 1)
                return;
            int i = fromIndex + 1;
            Feature toFeature = _Features[i];
            Feature fromFeature = _Features[fromIndex];
            _Features[i] = fromFeature;
            _Features[fromIndex] = toFeature;
        }

        public void MoveToTop(int fromIndex)
        {
            if (fromIndex < 0)
                throw new IndexOutOfRangeException();
            if (fromIndex > _Features.Count - 1)
                throw new IndexOutOfRangeException();
            if (fromIndex == 0)
                return;
            int i = fromIndex - 1;
            Feature fromFeature = _Features[fromIndex];
            _Features.RemoveAt(fromIndex);
            _Features.Insert(0, fromFeature);
        }

        public void ModeToBottom(int fromIndex)
        {
            if (fromIndex < 0)
                throw new IndexOutOfRangeException();
            if (fromIndex > _Features.Count - 1)
                throw new IndexOutOfRangeException();
            if (fromIndex == _Features.Count - 1)
                return;
            Feature fromFeature = _Features[fromIndex];
            _Features.RemoveAt(fromIndex);
            _Features.Add(fromFeature);
        }

        public USGS.Puma.NTS.Features.Feature GetFeature(int index)
        {
            return _Features[index];
        }

        public void UpdateFeature(USGS.Puma.NTS.Features.Feature feature, int index)
        {
            if (feature == null)
                throw new ArgumentNullException("The specified feature is null.");
            if (index < 0)
                throw new IndexOutOfRangeException();
            if (index > _Features.Count - 1)
                throw new IndexOutOfRangeException();
            if (!IsValidGeometry(feature.Geometry))
                throw new ArgumentException("Feature geometry type is incorrect.");
            _Features[index] = feature;
            _Extent = null;

        }

        public FeatureCollection GetFeatures()
        {
            FeatureCollection fc = new FeatureCollection();
            if (_Features.Count > 0)
            {
                for (int i = 0; i < _Features.Count; i++)
                {
                    fc.Add(_Features[i]);
                }
            }
            return fc;
        }

        public void Update()
        {
            _Extent = new Envelope();
            foreach (Feature f in _Features)
            {
                _Extent.ExpandToInclude(f.Geometry.EnvelopeInternal);
            }
        }

        public override GraphicLayerType LayerType
        {
            get { return GraphicLayerType.VectorLayer; }
        }

        public override int SRID
        {
            get
            {
                throw new NotImplementedException();
            }
            set
            {
                throw new NotImplementedException();
            }
        }

    }
}
