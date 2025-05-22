using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class ColorBar : System.Windows.Forms.Panel
    {
        private System.Drawing.Color[] _Colors = null;
        /// <summary>
        /// 
        /// </summary>
        public System.Drawing.Color[] Colors
        {
            get { return _Colors; }
            set 
            { 
                _Colors = value;
                this.Refresh();
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public ColorBar() : base()
        {
            
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="colorRamp"></param>
        /// <param name="sections"></param>
        public ColorBar(ColorRamp colorRamp, int sections)
            : this()
        {
            Colors = this.SampleColorRamp(colorRamp, sections);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="e"></param>
        protected override void OnPaint(System.Windows.Forms.PaintEventArgs e)
        {
            float dX;
            float dY;
            float x;
            float y;
            System.Drawing.Color color;
            float width = Convert.ToSingle(this.ClientRectangle.Width);
            float height = Convert.ToSingle(this.ClientRectangle.Height);
            
            base.OnPaint(e);
            if (_Colors != null)
            {
                if (_Colors.Length > 0)
                {
                    System.Drawing.SolidBrush brush = new System.Drawing.SolidBrush(System.Drawing.Color.Black);
                    if (height < width)
                    {
                        dX = width / Convert.ToSingle(_Colors.Length);
                        x = 0.0f;
                        y = 0.0f;
                        for (int i = 0; i < _Colors.Length; i++)
                        {
                            brush.Color = _Colors[i];
                            e.Graphics.FillRectangle(brush, x, y, dX, height);
                            x += dX;
                        }
                    }
                    else
                    {
                        dY = height / Convert.ToSingle(_Colors.Length);
                        x = 0.0f;
                        y = 0.0f;
                        for (int i = 0; i < _Colors.Length; i++)
                        {
                            brush.Color = _Colors[i];
                            e.Graphics.FillRectangle(brush, x, y, width, dY);
                            y += dY;
                        }
                    }
                    brush.Dispose();
                    brush = null;
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="colorRamp"></param>
        /// <param name="sections"></param>
        /// <returns></returns>
        public System.Drawing.Color[] SampleColorRamp(ColorRamp colorRamp, int sections)
        {
            System.Drawing.Color[] colors = new System.Drawing.Color[sections];

            float dX = 1.0f / Convert.ToSingle(sections);
            float f = dX / 2.0f;
            for (int i = 0; i < sections; i++)
            {
                colors[i] = colorRamp.GetColor(f);
                f += dX;
            }
            return colors;
        }

    }
}
