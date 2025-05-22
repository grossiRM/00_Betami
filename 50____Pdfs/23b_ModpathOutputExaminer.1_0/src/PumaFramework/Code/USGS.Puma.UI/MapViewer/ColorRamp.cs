using System;
using System.Collections.Generic;
using System.Text;
using System.Drawing;

namespace USGS.Puma.UI.MapViewer
{
    public class ColorRamp : IColorRamp
    {
        private float[] _Positions;

        private System.Drawing.Color[] _Colors;

        /// <summary>
        /// Gets or sets an array of colors that represents the colors to use at corresponding positions along a gradient.
        /// </summary>
        /// <value>An array of <see cref="System.Drawing.Color"/> structures that represents the colors to use at corresponding positions along a gradient.</value>
        /// <remarks>
        /// This property is an array of <see cref="System.Drawing.Color"/> structures that represents the colors to use at corresponding positions
        /// along a gradient. Along with the Positions property, this property defines a multicolor gradient.
        /// </remarks>
        public System.Drawing.Color[] Colors
        {
            get { return _Colors; }
            set 
            {
                if (value == null)
                {
                    _Colors = null;
                    _Positions = null;
                }
                else
                {
                    if (value.Length < 2)
                        throw new ArgumentException("The color ramp array must have at least two elements.");
                    _Colors = value;
                    ComputePositions();
                }
            }
        }


        internal ColorRamp() { }

        /// <summary>
        /// Initializes a new instance of the ColorBlend class.
        /// </summary>
        /// <param name="colors">An array of Color structures that represents the colors to use at corresponding positions along a gradient.</param>
        /// <param name="positions">An array of values that specify percentages of distance along the gradient line.</param>
        public ColorRamp(System.Drawing.Color[] colors)
        {
            Colors = colors;
        }

        /// <summary>
        /// Gets the color from the scale at position 'pos'.
        /// </summary>
        /// <remarks>If the position is outside the scale [0..1] only the fractional part
        /// is used (in other words the scale restarts for each integer-part).</remarks>
        /// <param name="pos">Position on scale between 0.0f and 1.0f</param>
        /// <returns>Color on scale</returns>
        public System.Drawing.Color GetColor(float position)
        {
            if (_Colors.Length != _Positions.Length)
                throw (new ArgumentException("Colors and Positions arrays must be of equal length"));
            if (_Colors.Length < 2)
                throw (new ArgumentException("At least two colors must be defined in the ColorBlend"));
            if (_Positions[0] != 0f)
                throw (new ArgumentException("First position value must be 0.0f"));
            if (_Positions[_Positions.Length - 1] != 1f)
                throw (new ArgumentException("Last position value must be 1.0f"));
            if (position > 1 || position < 0) position -= (float)Math.Floor(position);
            int i = 1;
            while (i < _Positions.Length && _Positions[i] < position)
                i++;
            float frac = (position - _Positions[i - 1]) / (_Positions[i] - _Positions[i - 1]);
            int R = (int)Math.Round((_Colors[i - 1].R * (1 - frac) + _Colors[i].R * frac));
            int G = (int)Math.Round((_Colors[i - 1].G * (1 - frac) + _Colors[i].G * frac));
            int B = (int)Math.Round((_Colors[i - 1].B * (1 - frac) + _Colors[i].B * frac));
            int A = (int)Math.Round((_Colors[i - 1].A * (1 - frac) + _Colors[i].A * frac));
            return Color.FromArgb(A, R, G, B);
        }

        private void ComputePositions()
        {
            if (_Colors == null)
                _Positions = null;
            if (_Colors.Length < 2)
            { return; }
            else if (_Colors.Length == 2)
            {
                _Positions = new float[2] { 0.0f, 1.0f };
            }
            else
            {
                _Positions = new float[_Colors.Length];
                int n = _Colors.Length - 1;
                float delta = 1.0f / Convert.ToSingle(n);
                _Positions[0] = 0.0f;
                _Positions[n] = 1.0f;
                for (int i = 1; i < n; i++)
                {
                    _Positions[i] = _Positions[i - 1] + delta;
                }
            }

        }
        #region Predefined color scales

        /// <summary>
        /// Gets a linear gradient scale with seven colours making a rainbow from red to violet.
        /// </summary>
        /// <remarks>
        /// Colors span the following with an interval of 1/6:
        /// { Color.Red, Color.Orange, Color.Yellow, Color.Green, Color.Blue, Color.Indigo, Color.Violet }
        /// </remarks>
        public static ColorRamp Rainbow7
        {
            get
            {
                return new ColorRamp(new Color[] { Color.Violet, Color.Indigo, Color.Blue, Color.Green, Color.Yellow, Color.Orange, Color.Red });
            }
        }

        /// <summary>
        /// Gets a linear gradient scale with five colours making a rainbow from red to blue.
        /// </summary>
        /// <remarks>
        /// Colors span the following with an interval of 0.25:
        /// { Color.Red, Color.Yellow, Color.Green, Color.Cyan, Color.Blue }
        /// </remarks>
        public static ColorRamp Rainbow5
        {
            get
            {
                return new ColorRamp(
                    new Color[] { Color.Blue, Color.Cyan, Color.Green, Color.Yellow, Color.Red });
            }
        }

        /// <summary>
        /// Gets a linear gradient scale from black to white
        /// </summary>
        public static ColorRamp BlackToWhite
        {
            get
            {
                return new ColorRamp(new Color[] { Color.Black, Color.White });
            }
        }

        /// <summary>
        /// Gets a linear gradient scale from white to black
        /// </summary>
        public static ColorRamp WhiteToBlack
        {
            get
            {
                return new ColorRamp(new Color[] { Color.White, Color.Black });
            }
        }

        /// <summary>
        /// Gets a linear gradient scale from red to green
        /// </summary>
        public static ColorRamp RedToGreen
        {
            get
            {
                return new ColorRamp(new Color[] { Color.Red, Color.Green });
            }
        }

        /// <summary>
        /// Gets a linear gradient scale from green to red
        /// </summary>
        public static ColorRamp GreenToRed
        {
            get
            {
                return new ColorRamp(new Color[] { Color.Green, Color.Red });
            }
        }

        /// <summary>
        /// Gets a linear gradient scale from blue to green
        /// </summary>
        public static ColorRamp BlueToGreen
        {
            get
            {
                return new ColorRamp(new Color[] { Color.Blue, Color.Green });
            }
        }

        /// <summary>
        /// Gets a linear gradient scale from green to blue
        /// </summary>
        public static ColorRamp GreenToBlue
        {
            get
            {
                return new ColorRamp(new Color[] { Color.Green, Color.Blue });
            }
        }

        /// <summary>
        /// Gets a linear gradient scale from red to blue
        /// </summary>
        public static ColorRamp RedToBlue
        {
            get
            {
                return new ColorRamp(new Color[] { Color.Red, Color.Blue });
            }
        }

        /// <summary>
        /// Gets a linear gradient scale from blue to red
        /// </summary>
        public static ColorRamp BlueToRed
        {
            get
            {
                return new ColorRamp(new Color[] { Color.Blue, Color.Red });
            }
        }

        #endregion

        #region Constructor helpers

        /// <summary>
        /// Creates a linear gradient scale from two colors
        /// </summary>
        /// <param name="fromColor"></param>
        /// <param name="toColor"></param>
        /// <returns></returns>
        public static ColorRamp TwoColors(System.Drawing.Color fromColor, System.Drawing.Color toColor)
        {
            return new ColorRamp(new Color[] { fromColor, toColor });
        }

        /// <summary>
        /// Creates a linear gradient scale from three colors
        /// </summary>
        public static ColorRamp ThreeColors(System.Drawing.Color fromColor, System.Drawing.Color middleColor, System.Drawing.Color toColor)
        {
            return new ColorRamp(new Color[] { fromColor, middleColor, toColor });
        }

        #endregion

    }
}
