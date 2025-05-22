using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class MapLegendSymbol : System.Windows.Forms.Panel
    {
        private System.Windows.Forms.Label labelText;

        /// <summary>
        /// 
        /// </summary>
        public MapLegendSymbol() : this(null, "") { }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="symbol"></param>
        public MapLegendSymbol(ISymbol symbol, string text)
        {
            InitializeComponent();
            SetData(symbol, text);
        }

        public void SetData(ISymbol symbol, string text)
        {
            if (symbol != null)
            {
                _Symbol = symbol;
            }
            else
            {
                _Symbol = null;
            }

            labelText.Text = "";
            if (!string.IsNullOrEmpty(text))
            {
                labelText.Text = text;
                
            }

        }

        private ISymbol _Symbol;
        /// <summary>
        /// 
        /// </summary>
        public ISymbol Symbol
        {
            get { return _Symbol; }
            set { _Symbol = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        public string LabelText
        {
            get { return labelText.Text; }
            set { labelText.Text = value; }
        }

        private void InitializeComponent()
        {
            this.labelText = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // labelText
            // 
            this.labelText.AutoSize = true;
            this.labelText.Location = new System.Drawing.Point(20, 0);
            this.labelText.Name = "labelText";
            this.labelText.Size = new System.Drawing.Size(0, 13);
            this.labelText.TabIndex = 0;
            // 
            // MapLegendSymbol
            // 
            this.Controls.Add(this.labelText);
            this.Height = 16;
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        protected override void OnPaint(System.Windows.Forms.PaintEventArgs e)
        {
            if (Symbol != null)
            {
                RendererHelper rh = new RendererHelper();
                rh.RenderLegendSymbol(Symbol, e.Graphics, new System.Drawing.Size(16, 16));
            }

            base.OnPaint(e);
        }
    }
}
