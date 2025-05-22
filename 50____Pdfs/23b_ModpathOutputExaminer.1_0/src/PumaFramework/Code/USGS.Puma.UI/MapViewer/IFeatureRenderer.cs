using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public enum SymbolType
    {
        Mixed = 0,
        PointSymbol = 1,
        LineSymbol= 2,
        FillSymbol = 3
    }
    public interface IFeatureRenderer : IRenderer
    {
        SymbolType SymbolType { get; }
    }
}
