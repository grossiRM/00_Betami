using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public enum FillStyle
    {
        Solid = 0,
        Hatch = 1
    }
    public interface IFillSymbol : ISymbol
    {
        ILineSymbol Outline { get; }
        bool EnableOutline { get; set; }
        bool OneColorForFillAndOutline { get; set; }
    }
}
