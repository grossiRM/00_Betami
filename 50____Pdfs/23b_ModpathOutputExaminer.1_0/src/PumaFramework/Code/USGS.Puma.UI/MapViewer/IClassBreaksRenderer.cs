using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    interface IClassBreaksRenderer : IFeatureRenderer
    {
        string RenderField { get; set; }

        bool EnableOutline { get; set; }

        void SetBreaks(double[] breaks);
        void SetBreaks(int[] breaks);
        void SetBreaks(float[] breaks);

        int BreakCount { get; }

        double GetBreak(int index);

        void SetSymbol(ISymbol symbol);
        void SetSymbols(ISymbol[] symbols);
        void SetSymbols(System.Drawing.Color[] colors);
        void SetSymbols(IColorRamp colorRamp);

        ISymbol GetSymbol(int index);

        int FindClassIndex(double value);
        int FindClassIndex(int value);
        int FindClassIndex(float value);
    }
}
