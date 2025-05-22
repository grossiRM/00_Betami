using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public interface ISingleSymbolRenderer : IFeatureRenderer
    {
        ISymbol Symbol { get; set; }
    }
}
