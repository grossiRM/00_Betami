using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public enum PointSymbolTypes
    {
        Circle = 0,
        Square = 1
    }

    public interface IPointSymbol : ISymbol
    {
        PointSymbolTypes SymbolType { get; set; }
        bool EnableOutline { get; set; }
        bool IsFilled { get; set; }
        System.Drawing.Color OutlineColor { get; set; }
        float Size { get; set; }
    }
}
