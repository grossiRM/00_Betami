using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using USGS.Puma.Modpath;
using USGS.Puma.Modpath.IO;
using USGS.Puma.UI;
using USGS.Puma.UI.MapViewer;
using USGS.Puma.Utilities;
using USGS.Puma.NTS;
using USGS.Puma.NTS.Features;
using GeoAPI.Geometries;

namespace ModpathOutputExaminer
{
    public enum EndpointRenderOptionTypes
    {
        SingleSymbol = 0,
        UniqueValues = 1,
        ContinuousValues = 2
    }

    public partial class EndpointSymbologyDialog : Form
    {
        public EndpointSymbologyDialog()
        {
            InitializeComponent();

            _UniqueValueSeeds.Add(100);
            _UniqueValueSeeds.Add(300);
            _UniqueValueSeeds.Add(1500);

            EndpointOption = EndpointLocationTypes.InitialPoint;
            SingleSymbolColor = System.Drawing.Color.CadetBlue;
            SymbolType = PointSymbolTypes.Circle;
            SymbolSize = 2.0f;
            RenderOption = EndpointRenderOptionTypes.SingleSymbol;
            UniqueValuesRenderField = "FinalZone";
            ContinuousValuesRenderField = "TravelTime";
            ExcludeZone1 = false;
            UniqueValuesPaletteIndex = 1;
            ContinuousValuesPalleteIndex = 1;

            Color[] colors = colorBarRainbow7.SampleColorRamp(ColorRamp.Rainbow7, 100);
            colorBarRainbow7.Colors = colors;
            colors = colorBarRainbow5.SampleColorRamp(ColorRamp.Rainbow5, 100);
            colorBarRainbow5.Colors = colors;
            colors = colorBarRedToGreen.SampleColorRamp(ColorRamp.RedToGreen, 100);
            colorBarRedToGreen.Colors = colors;
            colors = colorBarRedToBlue.SampleColorRamp(ColorRamp.RedToBlue, 100);
            colorBarRedToBlue.Colors = colors;
            colors = colorBarBlueToGreen.SampleColorRamp(ColorRamp.BlueToGreen, 100);
            colorBarBlueToGreen.Colors = colors;

            colors = NumericValueRenderer.GenerateRandomColors(15, _UniqueValueSeeds[0]);
            colorBarUvPalette1.Colors = colors;
            colors = NumericValueRenderer.GenerateRandomColors(15, _UniqueValueSeeds[1]);
            colorBarUvPalette2.Colors = colors;
            colors = NumericValueRenderer.GenerateRandomColors(15, _UniqueValueSeeds[2]);
            colorBarUvPalette3.Colors = colors;

            cboSymbolType.Items.Add("Circle");
            cboSymbolType.Items.Add("Square");

            cboUniqueValuesRenderField.Items.Add("FinalZone");
            cboUniqueValuesRenderField.Items.Add("InitZone");
            cboUniqueValuesRenderField.Items.Add("Group");
            cboUniqueValuesRenderField.Items.Add("Status");

            cboContinuousRenderField.Items.Add("TravelTime");

            cboEndpointOption.Items.Add("Plot initial locations");
            cboEndpointOption.Items.Add("Plot final locations");

            cboSize.Items.Add("1");
            cboSize.Items.Add("2");
            cboSize.Items.Add("3");
            cboSize.Items.Add("4");
            cboSize.Items.Add("5");

        }

        private List<int> _UniqueValueSeeds = new List<int>();

        #region Public Properties
        private EndpointLocationTypes _EndpointOption;
        /// <summary>
        /// 
        /// </summary>
        public EndpointLocationTypes EndpointOption
        {
            get { return _EndpointOption; }
            set { _EndpointOption = value; }
        }


        private System.Drawing.Color _SingleSymbolColor;
        /// <summary>
        /// 
        /// </summary>
        public System.Drawing.Color SingleSymbolColor
        {
            get { return _SingleSymbolColor; }
            set { _SingleSymbolColor = value; }
        }

        private PointSymbolTypes _SymbolType;
        /// <summary>
        /// 
        /// </summary>
        public PointSymbolTypes SymbolType
        {
            get { return _SymbolType; }
            set { _SymbolType = value; }
        }

        private float _SymbolSize;
        /// <summary>
        /// 
        /// </summary>
        public float SymbolSize
        {
            get { return _SymbolSize; }
            set { _SymbolSize = value; }
        }

        private EndpointRenderOptionTypes _RenderOption;
        /// <summary>
        /// 
        /// </summary>
        public EndpointRenderOptionTypes RenderOption
        {
            get { return _RenderOption; }
            set { _RenderOption = value; }
        }

        private string _UniqueValuesRenderField;
        /// <summary>
        /// 
        /// </summary>
        public string UniqueValuesRenderField
        {
            get { return _UniqueValuesRenderField; }
            set { _UniqueValuesRenderField = value; }
        }

        private string _ContinuousValuesRenderField;
        /// <summary>
        /// 
        /// </summary>
        public string ContinuousValuesRenderField
        {
            get { return _ContinuousValuesRenderField; }
            set { _ContinuousValuesRenderField = value; }
        }

        private int _UniqueValuesPaletteIndex;
        /// <summary>
        /// 
        /// </summary>
        public int UniqueValuesPaletteIndex
        {
            get { return _UniqueValuesPaletteIndex; }
            set { _UniqueValuesPaletteIndex = value; }
        }

        private int _ContinuousValuesPalleteIndex;
        /// <summary>
        /// 
        /// </summary>
        public int ContinuousValuesPalleteIndex
        {
            get { return _ContinuousValuesPalleteIndex; }
            set { _ContinuousValuesPalleteIndex = value; }
        }

        private bool _ExcludeZone1;
        /// <summary>
        /// 
        /// </summary>
        public bool ExcludeZone1
        {
            get { return _ExcludeZone1; }
            set { _ExcludeZone1 = value; }
        }

        #endregion

        #region Event Handlers
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void buttonSelectColor_Click(object sender, EventArgs e)
        {
            ColorDialog dialog = new ColorDialog();
            dialog.Color = panelSingleSymbolColor.BackColor;
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                panelSingleSymbolColor.BackColor = dialog.Color;
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void buttonOK_Click(object sender, EventArgs e)
        {
            SingleSymbolColor = panelSingleSymbolColor.BackColor;
            this.DialogResult = DialogResult.OK;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void buttonCancel_Click(object sender, EventArgs e)
        {
            this.DialogResult = DialogResult.Cancel;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void colorBarUvPalette1_Click(object sender, EventArgs e)
        {
            radioBtnUvPalette1.Checked = true;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void colorBarUvPalette2_Click(object sender, EventArgs e)
        {
            radioBtnUvPalette2.Checked = true;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void colorBarUvPalette3_Click(object sender, EventArgs e)
        {
            radioBtnUvPalette3.Checked = true;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void colorBarRainbow7_Click(object sender, EventArgs e)
        {
            radioBtnRainbow7.Checked = true;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void colorBarRainbow5_Click(object sender, EventArgs e)
        {
            radioBtnRainbow5.Checked = true;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void colorBarRedToGreen_Click(object sender, EventArgs e)
        {
            radioBtnRedToGreen.Checked = true;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void colorBarRedToBlue_Click(object sender, EventArgs e)
        {
            radioBtnRedToBlue.Checked = true;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void colorBarBlueToGreen_Click(object sender, EventArgs e)
        {
            radioBtnBlueToGreen.Checked = true;
        }

        #endregion

        #region Public Methods
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public bool ShowMapSymbologyDialog()
        {
            #region Initialize dialog data
            switch (EndpointOption)
            {
                case EndpointLocationTypes.InitialPoint:
                    cboEndpointOption.SelectedIndex = 0;
                    break;
                case EndpointLocationTypes.FinalPoint:
                    cboEndpointOption.SelectedIndex = 1;
                    break;
                default:
                    cboEndpointOption.SelectedIndex = 0;
                    break;
            }

            switch (SymbolType)
            {
                case PointSymbolTypes.Circle:
                    cboSymbolType.SelectedIndex = 0;
                    break;
                case PointSymbolTypes.Square:
                    cboSymbolType.SelectedIndex = 1;
                    break;
                default:
                    cboSymbolType.SelectedIndex = 0;
                    break;
            }
            cboSize.Text = SymbolSize.ToString();
            panelSingleSymbolColor.BackColor = SingleSymbolColor;
            switch (RenderOption)
            {
                case EndpointRenderOptionTypes.SingleSymbol:
                    radioBtnSingleSymbol.Checked = true;
                    break;
                case EndpointRenderOptionTypes.UniqueValues:
                    radioBtnUniqueValues.Checked = true;
                    break;
                case EndpointRenderOptionTypes.ContinuousValues:
                    radioBtnContinuousValues.Checked = true;
                    break;
                default:
                    break;
            }

            if (UniqueValuesRenderField == "FinalZone")
            {
                cboUniqueValuesRenderField.SelectedIndex = 0;
            }
            else if (UniqueValuesRenderField == "InitialZone")
            {
                cboUniqueValuesRenderField.SelectedIndex = 1;
            }
            else if (UniqueValuesRenderField == "Group")
            {
                cboUniqueValuesRenderField.SelectedIndex = 2;
            }
            else
            {
                cboUniqueValuesRenderField.SelectedIndex = 3;
            }

            if (ContinuousValuesRenderField == "TraveTime")
            {
                cboContinuousRenderField.SelectedIndex = 0;
            }
            else
            {
                cboContinuousRenderField.SelectedIndex = 0;
            }

            if (UniqueValuesPaletteIndex == 0)
            {
                radioBtnUvPalette1.Checked = true;
            }
            else if (UniqueValuesPaletteIndex == 1)
            {
                radioBtnUvPalette2.Checked = true;
            }
            else
            {
                radioBtnUvPalette3.Checked = true;
            }

            if (ContinuousValuesPalleteIndex == 0)
            {
                radioBtnRainbow7.Checked = true;
            }
            else if (ContinuousValuesPalleteIndex == 1)
            {
                radioBtnRainbow5.Checked = true;
            }
            else if (ContinuousValuesPalleteIndex == 2)
            {
                radioBtnRedToGreen.Checked = true;
            }
            else if (ContinuousValuesPalleteIndex == 3)
            {
                radioBtnRedToBlue.Checked = true;
            }
            else
            {
                radioBtnBlueToGreen.Checked = true;
            }
            #endregion

            // Show dialog
            if (this.ShowDialog() == DialogResult.OK)
            {
                if (cboEndpointOption.SelectedIndex == 1)
                { EndpointOption = EndpointLocationTypes.FinalPoint; }
                else
                { EndpointOption = EndpointLocationTypes.InitialPoint;}

                if (cboSymbolType.SelectedIndex == 0)
                {
                    SymbolType = PointSymbolTypes.Circle;
                }
                else
                {
                    SymbolType = PointSymbolTypes.Square;
                }
                SymbolSize = Single.Parse(cboSize.Text);
                SingleSymbolColor = panelSingleSymbolColor.BackColor;
                
                if (radioBtnSingleSymbol.Checked)
                {
                    RenderOption = EndpointRenderOptionTypes.SingleSymbol;
                }
                else if (radioBtnUniqueValues.Checked)
                {
                    RenderOption = EndpointRenderOptionTypes.UniqueValues;
                }
                else if (radioBtnContinuousValues.Checked)
                {
                    RenderOption = EndpointRenderOptionTypes.ContinuousValues;
                }

                UniqueValuesRenderField = cboUniqueValuesRenderField.Text;
                if (radioBtnUvPalette1.Checked)
                { UniqueValuesPaletteIndex = 0; }
                else if (radioBtnUvPalette2.Checked)
                { UniqueValuesPaletteIndex = 1; }
                else if (radioBtnUvPalette3.Checked)
                { UniqueValuesPaletteIndex = 2; }

                ContinuousValuesRenderField = cboContinuousRenderField.Text;
                if (radioBtnRainbow7.Checked)
                { ContinuousValuesPalleteIndex = 0; }
                else if (radioBtnRainbow5.Checked)
                { ContinuousValuesPalleteIndex = 1; }
                else if (radioBtnRedToGreen.Checked)
                { ContinuousValuesPalleteIndex = 2; }
                else if (radioBtnRedToBlue.Checked)
                { ContinuousValuesPalleteIndex = 3; }
                else if (radioBtnBlueToGreen.Checked)
                { ContinuousValuesPalleteIndex = 4; }

                return true;
            }
            return false;
        }

        public void SetRenderer(FeatureLayer layer)
        {
            IFeatureRenderer renderer;

            if (layer == null)
            { throw new ArgumentNullException("The specified layer object is not defined."); }

            if (layer.GeometryType != LayerGeometryType.Point)
            { throw new ArgumentException("The layer does not contain point features."); }
            
            SimplePointSymbol pointSymbol;
            switch (SymbolType)
            {
                case PointSymbolTypes.Circle:
                    pointSymbol = new SimplePointSymbol(PointSymbolTypes.Circle, SingleSymbolColor, SymbolSize);
                    break;
                case PointSymbolTypes.Square:
                    pointSymbol = new SimplePointSymbol(PointSymbolTypes.Square, SingleSymbolColor, SymbolSize);
                    break;
                default:
                    pointSymbol = new SimplePointSymbol(PointSymbolTypes.Circle, SingleSymbolColor, SymbolSize);
                    break;
            }
            pointSymbol.OutlineColor = pointSymbol.Color;
            pointSymbol.EnableOutline = true;
            pointSymbol.IsFilled = true;

            switch (RenderOption)
            {
                case EndpointRenderOptionTypes.SingleSymbol:
                    renderer = new SingleSymbolRenderer(pointSymbol as IPointSymbol);
                    layer.Renderer = renderer;
                    return;

                case EndpointRenderOptionTypes.UniqueValues:
                    renderer = CreateUniqueValuePointRenderer(layer.GetFeatures(), UniqueValuesRenderField, pointSymbol, _UniqueValueSeeds[UniqueValuesPaletteIndex]);
                    if (ExcludeZone1)
                    {
                        (renderer as NumericValueRenderer).ExcludedValues.Add(1.0);
                    }
                    layer.Renderer = renderer;
                    return;

                case EndpointRenderOptionTypes.ContinuousValues:
                    ColorRampRenderer crRenderer = new ColorRampRenderer(USGS.Puma.UI.MapViewer.SymbolType.PointSymbol);
                    crRenderer.BaseSymbol = pointSymbol;
                    crRenderer.RenderField = ContinuousValuesRenderField;
                    
                    switch (ContinuousValuesPalleteIndex)
                    {
                        case 0:
                            crRenderer.ColorRamp = ColorRamp.Rainbow7;
                            break;
                        case 1:
                            crRenderer.ColorRamp = ColorRamp.Rainbow5;
                            break;
                        case 2:
                            crRenderer.ColorRamp = ColorRamp.RedToGreen;
                            break;
                        case 3:
                            crRenderer.ColorRamp = ColorRamp.RedToBlue;
                            break;
                        case 4:
                            crRenderer.ColorRamp = ColorRamp.BlueToGreen;
                            break;
                        default:
                            break;
                    }

                    FeatureCollection features = layer.GetFeatures();
                    double[] values = new double[features.Count];
                    for (int i = 0; i < features.Count; i++)
                    {
                        values[i] = Convert.ToDouble(features[i].Attributes[ContinuousValuesRenderField]);
                    }
                    IArrayUtility<double> autil = new ArrayUtility() as IArrayUtility<double>;
                    double minValue;
                    double maxValue;
                    autil.FindMinimumAndMaximum(values, out minValue, out maxValue);
                    crRenderer.MinimumValue = minValue;
                    crRenderer.MaximumValue = maxValue;

                    layer.Renderer = crRenderer as IFeatureRenderer;
                    return;
            }
        }
        #endregion

        #region Private Methods
        private IFeatureRenderer CreateUniqueValuePointRenderer(FeatureCollection features, string renderField, IPointSymbol symbolTemplate, int seed)
        {
            double[] zones = new double[features.Count];
            for (int i = 0; i < features.Count; i++)
            {
                zones[i] = Convert.ToDouble(features[i].Attributes[renderField]);
            }
            IArrayUtility<double> autil = new ArrayUtility();
            List<double> uvList = autil.GetUniqueValues(zones);
            uvList.Sort();
            double[] uvalues = uvList.ToArray<double>();
            ISymbol[] pointSymbol = new ISymbol[uvList.Count];
            for (int i = 0; i < uvList.Count; i++)
            {
                SimplePointSymbol ptSymbol = new SimplePointSymbol();
                ptSymbol.SymbolType = symbolTemplate.SymbolType;
                ptSymbol.Size = symbolTemplate.Size;
                ptSymbol.IsFilled = symbolTemplate.IsFilled;
                ptSymbol.EnableOutline = symbolTemplate.EnableOutline;
                pointSymbol[i] = ptSymbol;
            }

            NumericValueRenderer uvRenderer = new NumericValueRenderer(USGS.Puma.UI.MapViewer.SymbolType.PointSymbol, uvalues, pointSymbol, seed);
            for (int i = 0; i < uvRenderer.ValueCount; i++)
            {
                SimplePointSymbol sym = uvRenderer.Symbols[i] as SimplePointSymbol;
                sym.OutlineColor = sym.Color;
            }
            uvRenderer.RenderField = renderField;

            return uvRenderer as IFeatureRenderer;

        }

        #endregion

    }
}
