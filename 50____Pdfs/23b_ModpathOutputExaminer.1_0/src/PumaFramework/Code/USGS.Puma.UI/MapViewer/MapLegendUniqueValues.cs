using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class MapLegendUniqueValues : MapLegendItemData
    {
        #region Private Fields
        private System.Windows.Forms.Label labelRenderField;
        private System.Windows.Forms.Panel panelHeader;
        private System.Windows.Forms.Panel panelItems;
        #endregion

        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        public MapLegendUniqueValues() : this(null) { }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="renderer"></param>
        public MapLegendUniqueValues(NumericValueRenderer renderer)
        {
            InitializeComponent();

            Update(renderer);

        }
        #endregion

        #region Public Methods
        /// <summary>
        /// 
        /// </summary>
        /// <param name="renderer"></param>
        public void Update(NumericValueRenderer renderer)
        {
            if (renderer != null)
            {
                labelRenderField.Text = renderer.RenderField;
                panelItems.Controls.Clear();
                int y = 0;
                double v = 0.0;
                for (int i = 0; i < renderer.ValueCount; i++)
                {
                    v = renderer.GetValue(i);
                    MapLegendSymbol item = new MapLegendSymbol(renderer.Symbols[i], v.ToString());
                    item.Location = new System.Drawing.Point(0, y);
                    item.Width = panelItems.Width - 50;
                    panelItems.Controls.Add(item);
                    y += item.Height;
                }
                panelItems.Height = y;
                this.Height = panelHeader.Height + panelItems.Height + 2;
            }

        }

        #endregion

        #region Private Methods
        private void InitializeComponent()
        {
            this.labelRenderField = new System.Windows.Forms.Label();
            this.panelItems = new System.Windows.Forms.Panel();
            this.panelHeader = new System.Windows.Forms.Panel();
            this.SuspendLayout();
            // 
            // labelRenderField
            // 
            this.labelRenderField.AutoSize = true;
            this.labelRenderField.Location = new System.Drawing.Point(16, 0);
            this.labelRenderField.Name = "labelRenderField";
            this.labelRenderField.Size = new System.Drawing.Size(0, 13);
            this.labelRenderField.TabIndex = 0;
            // 
            // panelItems
            // 
            this.panelItems.BackColor = System.Drawing.Color.White;
            this.panelItems.Dock = System.Windows.Forms.DockStyle.Top;
            this.panelItems.Location = new System.Drawing.Point(0, 0);
            this.panelItems.Name = "panelItems";
            this.panelItems.Size = new System.Drawing.Size(200, 100);
            this.panelItems.TabIndex = 0;
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
            // MapLegendUniqueValues
            // 
            this.panelHeader.Controls.Add(this.labelRenderField);
            this.Controls.Add(this.panelItems);
            this.Controls.Add(this.panelHeader);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
    }
}
