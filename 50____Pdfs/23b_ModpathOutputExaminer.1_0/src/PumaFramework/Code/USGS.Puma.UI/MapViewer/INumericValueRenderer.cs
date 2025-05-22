using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public interface INumericValueRenderer : IFeatureRenderer
    {
        string RenderField { get; set; }

        int ValueCount { get; }

        void AddValue(int value);
        void AddValue(int value, ISymbol symbol);
        void AddValue(float value);
        void AddValue(float value, ISymbol symbol);
        void AddValue(double value);
        void AddValue(double value, ISymbol symbol);

        int IndexOf(int value);
        int IndexOf(float value);
        int IndexOf(double value);

        ISymbol GetSymbol(int index);

        double GetValue(int index);

        double[] Values { get; }

        List<double> ExcludedValues { get; }

        ISymbol[] Symbols { get; }

        void Clear();

        void RemoveAt(int index);
    }
}
