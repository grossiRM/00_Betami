using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public interface IColorRampRenderer : IFeatureRenderer
    {
        string RenderField { get; set; }
        IColorRamp ColorRamp { get; set; }
        ISymbol BaseSymbol { get; set; }
        double MinimumValue { get; set; }
        double MaximumValue { get; set; }
        float GetPosition(double value);
    }
}
