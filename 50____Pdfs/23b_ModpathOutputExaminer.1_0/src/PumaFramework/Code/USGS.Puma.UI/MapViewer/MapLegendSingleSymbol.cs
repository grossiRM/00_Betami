using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class MapLegendSingleSymbol : MapLegendItemData
    {
        #region Private Fields
        private System.Windows.Forms.Panel panelItems;
        private MapLegendSymbol _DataItem;
        #endregion

        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        public MapLegendSingleSymbol() : this(null) { }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="renderer"></param>
        public MapLegendSingleSymbol(SingleSymbolRenderer renderer)
        {
            InitializeComponent();

            // Add code to initialize renderer
            if (renderer != null)
            {
                _DataItem = new MapLegendSymbol(renderer.Symbol, "");
                panelItems.Controls.Add(_DataItem);
                panelItems.Height = _DataItem.Height;
                this.Height = panelItems.Height;
            }
        }
        #endregion

        #region Private Methods
        private void InitializeComponent()
        {
            this.panelItems = new System.Windows.Forms.Panel();
            this.SuspendLayout();
            // 
            // panelItems
            // 
            this.panelItems.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left)
                        | System.Windows.Forms.AnchorStyles.Right)));
            this.panelItems.Location = new System.Drawing.Point(0, 0);
            this.panelItems.Name = "panelItems";
            this.panelItems.Size = new System.Drawing.Size(200, 100);
            this.panelItems.TabIndex = 0;
            // 
            // MapLegendSingleSymbol
            // 
            this.Controls.Add(this.panelItems);
            this.ResumeLayout(false);

        }
        #endregion
    }
}
