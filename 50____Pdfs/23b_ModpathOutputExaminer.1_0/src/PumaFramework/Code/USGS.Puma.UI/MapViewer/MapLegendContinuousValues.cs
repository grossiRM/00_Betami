using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class MapLegendContinuousValues : MapLegendItemData
    {
        #region Private Fields
        private System.Windows.Forms.Label labelRenderField;
        private System.Windows.Forms.Panel panelBody;
        private ColorBar colorBarData;
        private System.Windows.Forms.Panel panelMinValue;
        private System.Windows.Forms.Panel panelMaxValue;
        private System.Windows.Forms.Label labelMinValue;
        private System.Windows.Forms.Label labelMaxValue;
        private System.Windows.Forms.Panel panelHeader;
        #endregion

        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        public MapLegendContinuousValues() : this(null) { }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="renderer"></param>
        public MapLegendContinuousValues(ColorRampRenderer renderer) : this(renderer, 150) { }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="renderer"></param>
        public MapLegendContinuousValues(ColorRampRenderer renderer, int height)
        {
            InitializeComponent();

            this.panelBody.Height = height;
            this.Controls.Add(this.panelBody);
            this.Controls.Add(this.panelHeader);
            this.panelMaxValue.Controls.Add(this.labelMaxValue);
            this.panelMinValue.Controls.Add(this.labelMinValue);
            this.Height = panelHeader.Height + panelBody.Height;
            this.panelHeader.Controls.Add(this.labelRenderField);
            this.panelBody.Controls.Add(this.panelMinValue);
            this.panelBody.Controls.Add(this.panelMaxValue);
            this.panelBody.Controls.Add(this.colorBarData);

            Update(renderer);

        }
        #endregion

        #region Private Methods
        /// <summary>
        /// 
        /// </summary>
        private void InitializeComponent()
        {
            this.panelHeader = new System.Windows.Forms.Panel();
            this.labelRenderField = new System.Windows.Forms.Label();
            this.panelBody = new System.Windows.Forms.Panel();
            this.panelMinValue = new System.Windows.Forms.Panel();
            this.panelMaxValue = new System.Windows.Forms.Panel();
            this.colorBarData = new USGS.Puma.UI.MapViewer.ColorBar();
            this.labelMinValue = new System.Windows.Forms.Label();
            this.labelMaxValue = new System.Windows.Forms.Label();
            this.panelBody.SuspendLayout();
            this.SuspendLayout();
            // 
            // panelHeader
            // 
            this.panelHeader.BackColor = System.Drawing.Color.White;
            this.panelHeader.Dock = System.Windows.Forms.DockStyle.Top;
            this.panelHeader.Location = new System.Drawing.Point(0, 0);
            this.panelHeader.Name = "panelHeader";
            this.panelHeader.Size = new System.Drawing.Size(200, 20);
            this.panelHeader.TabIndex = 0;
            // 
            // labelRenderField
            // 
            this.labelRenderField.AutoSize = true;
            this.labelRenderField.Location = new System.Drawing.Point(20, 0);
            this.labelRenderField.Name = "labelRenderField";
            this.labelRenderField.Size = new System.Drawing.Size(100, 23);
            this.labelRenderField.TabIndex = 0;
            // 
            // panelBody
            // 
            this.panelBody.BackColor = System.Drawing.Color.White;
            this.panelBody.Dock = System.Windows.Forms.DockStyle.Top;
            this.panelBody.Location = new System.Drawing.Point(0, 0);
            this.panelBody.Name = "panelBody";
            this.panelBody.Size = new System.Drawing.Size(200, 150);
            this.panelBody.TabIndex = 0;
            // 
            // panelMinValue
            // 
            this.panelMinValue.BackColor = System.Drawing.Color.White;
            this.panelMinValue.Dock = System.Windows.Forms.DockStyle.Top;
            this.panelMinValue.Location = new System.Drawing.Point(16, 0);
            this.panelMinValue.Name = "panelMinValue";
            this.panelMinValue.Size = new System.Drawing.Size(184, 20);
            this.panelMinValue.TabIndex = 0;
            // 
            // panelMaxValue
            // 
            this.panelMaxValue.BackColor = System.Drawing.Color.White;
            this.panelMaxValue.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.panelMaxValue.Location = new System.Drawing.Point(16, 130);
            this.panelMaxValue.Name = "panelMaxValue";
            this.panelMaxValue.Size = new System.Drawing.Size(184, 16);
            this.panelMaxValue.TabIndex = 0;
            // 
            // colorBarData
            // 
            this.colorBarData.Colors = null;
            this.colorBarData.Dock = System.Windows.Forms.DockStyle.Left;
            this.colorBarData.Location = new System.Drawing.Point(0, 0);
            this.colorBarData.Name = "colorBarData";
            this.colorBarData.Size = new System.Drawing.Size(16, 150);
            this.colorBarData.TabIndex = 0;
            // 
            // labelMinValue
            // 
            this.labelMinValue.AutoSize = true;
            this.labelMinValue.Location = new System.Drawing.Point(5, 0);
            this.labelMinValue.Name = "labelMinValue";
            this.labelMinValue.Size = new System.Drawing.Size(100, 13);
            this.labelMinValue.TabIndex = 0;
            this.labelMinValue.Text = "label1";
            // 
            // labelMaxValue
            // 
            this.labelMaxValue.AutoSize = true;
            this.labelMaxValue.Location = new System.Drawing.Point(5, 4);
            this.labelMaxValue.Name = "labelMaxValue";
            this.labelMaxValue.Size = new System.Drawing.Size(100, 13);
            this.labelMaxValue.TabIndex = 0;
            this.labelMaxValue.Text = "label1";
            this.panelBody.ResumeLayout(false);
            this.ResumeLayout(false);

        }
        #endregion

        #region Public Methods
        public void Update(ColorRampRenderer renderer)
        {
            if (renderer != null)
            {
                System.Drawing.Color[] colors = colorBarData.SampleColorRamp(renderer.ColorRamp as ColorRamp, 100);
                colorBarData.Colors = colors;
                labelRenderField.Text = renderer.RenderField;
                labelMinValue.Text = renderer.MinimumValue.ToString("0.000000E+00");
                labelMaxValue.Text = renderer.MaximumValue.ToString("0.000000E+00");
            }
        }
        #endregion
    }
}
