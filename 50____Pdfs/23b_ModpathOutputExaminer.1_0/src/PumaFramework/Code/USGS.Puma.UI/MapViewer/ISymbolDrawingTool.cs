using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public interface ISymbolDrawingTool
    {
        System.Drawing.Brush FillBrush { get; set; }
        System.Drawing.Pen Pen { get; set; }
        bool EnableOutline { get; set; }
        ISymbol Symbol { get; set; }
    }
}
