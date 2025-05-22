using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class ColorHelper
    {
        #region Private Fields
        private Random _Random = null;
        #endregion

        #region Public Methods
        public System.Drawing.Color GetRandomColor()
        {
            int r = GenerateRandomColorComponent();
            int g = GenerateRandomColorComponent();
            int b = GenerateRandomColorComponent();
            return System.Drawing.Color.FromArgb(r, g, b);
        }
        public System.Drawing.Color[] GenerateRandomColorArray(int valueCount, int seed)
        {
            if (valueCount < 1)
            { return null; }

            _Random = new Random(seed);

            System.Drawing.Color[] colors = new System.Drawing.Color[valueCount];
            for (int i = 0; i < colors.Length; i++)
            {
                colors[i] = GetRandomColor();
            }

            return colors;

        }
        public System.Drawing.Color[] GenerateRandomColorArray(int valueCount)
        {
            if (valueCount < 1)
            { return null; }

            System.Drawing.Color[] colors = new System.Drawing.Color[valueCount];
            for (int i = 0; i < colors.Length; i++)
            {
                colors[i] = GetRandomColor();
            }

            return colors;

        }
        #endregion

        #region Private Methods
        private int GenerateRandomColorComponent()
        {
            if (_Random == null)
                _Random = new Random();
            return Convert.ToInt32(Math.Round(255.0 * _Random.NextDouble(), 0));
        }
        #endregion

    }
}
