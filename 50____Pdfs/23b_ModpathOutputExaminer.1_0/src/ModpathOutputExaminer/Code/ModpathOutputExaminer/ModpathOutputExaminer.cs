using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using USGS.Puma.Core;
using USGS.Puma.Utilities;
using USGS.Puma.FiniteDifference;
using USGS.Puma.IO;
using USGS.Puma.UI;
using USGS.Puma.UI.MapViewer;
using USGS.Puma.Modflow;
using USGS.Puma.Modpath;
using USGS.Puma.Modpath.IO;
using USGS.Puma.UI.Modpath;
using USGS.Puma.NTS;
using USGS.Puma.NTS.Features;
using USGS.Puma.NTS.Geometries;
using GeoAPI.Geometries;
using USGS.ModflowTrainingTools;

namespace ModpathOutputExaminer
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public enum ModpathSimulationType
    {
        /// <summary>
        /// 
        /// </summary>
        None = 0,
        /// <summary>
        /// 
        /// </summary>
        Endpoint = 1,
        /// <summary>
        /// 
        /// </summary>
        Pathline = 2,
        /// <summary>
        /// 
        /// </summary>
        Timeseries = 3
    }

    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public partial class ModpathOutputExaminer : Form
    {
        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="ModpathOutputExaminer"/> class.
        /// </summary>
        /// <param name="args">The args.</param>
        /// <remarks></remarks>
        public ModpathOutputExaminer(string[] args)
        {
            InitializeComponent();

            // Remove unused tab pages from the tabMapInfo control.
            tabMapInfo.TabPages.Remove(tabPageMapInfoInformation);

            // Create and initialize mapControl, then add to the mapPanel
            mapControl = CreateAndInitializeMapControl();
            mapControl.Dock = DockStyle.Fill;
            mapPanel.Panel1.Controls.Add(mapControl);
            legendParticles.AutoScroll = true;
            legendParticles.LayerVisibilityChanged += new EventHandler<EventArgs>(legendParticles_LayerVisibilityChanged);

            // Create MapControl cursors
            _ReCenterCursor = MapControl.CreateCursor(MapControlCursor.ReCenter);
            _ZoomInCursor = MapControl.CreateCursor(MapControlCursor.ZoomIn);
            _ZoomOutCursor = MapControl.CreateCursor(MapControlCursor.ZoomOut);
            _ActiveTool = ActiveTool.Pointer;

            // Initialize the map tip object
            _MapTip = new ToolTip();
            _MapTip.ShowAlways = true;

            // Create and initialize indexMapControl, then add to the map panel container
            indexMapControl = CreateAndInitializeIndexMapControl(mapControl);
            indexMapControl.Dock = DockStyle.Fill;
            splitConMapviewerUtility.Panel2.Controls.Add(indexMapControl);

            toolStripComboBoxQuickView.Items.Clear();
            toolStripButtonContinuousLoop.Checked = true;
            for (int i = 0; i < 10; i++)
            {
                double v = (100 * (i + 1)) / 1000.0;
                toolStripComboBoxAnimationInterval.Items.Add(v);
            }
            toolStripComboBoxAnimationInterval.SelectedIndex = 0;
            EnableTimePointNavigationControls(false, false);

            SetMapMenuAndToolbarAccess(false);

            toolStripStatusMessage.Text = "No dataset is loaded.";
            tabDataset.Enabled = false;
            toolStripMain.Enabled = false;

            // Turn on basemap visibility
            SetBasemapVisibility(true);

            // This will clear the map legend panel because no dataset is loaded.
            BuildMapLegend();

            // Check the command line arguments and open a dataset if a simulation
            // file was specified.
            if (args != null)
            {
                if (args.Length > 0)
                {
                    if (!string.IsNullOrEmpty(args[0]))
                    {
                        OpenDataset(args[0]);
                    }
                }
            }

        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:System.Windows.Forms.Form"/> class.
        /// </summary>
        /// <remarks></remarks>
        public ModpathOutputExaminer() : this(null) { }
        #endregion

        #region Enumerations
        /// <summary>
        /// 
        /// </summary>
        /// <remarks></remarks>
        public enum ParticleView
        {
            /// <summary>
            /// 
            /// </summary>
            Summary = 0,
            /// <summary>
            /// 
            /// </summary>
            Records = 1,
            /// <summary>
            /// 
            /// </summary>
            Preview = 2
        }
        /// <summary>
        /// 
        /// </summary>
        /// <remarks></remarks>
        public enum ActiveTool
        {
            /// <summary>
            /// 
            /// </summary>
            Pointer = 0,
            /// <summary>
            /// 
            /// </summary>
            ReCenter = 1,
            /// <summary>
            /// 
            /// </summary>
            ZoomIn = 2,
            /// <summary>
            /// 
            /// </summary>
            ZoomOut = 3,
            /// <summary>
            /// 
            /// </summary>
            DefinePolygon = 4,
            /// <summary>
            /// 
            /// </summary>
            DefineRectangle = 5,
            /// <summary>
            /// 
            /// </summary>
            DefineLineString = 6,
            /// <summary>
            /// 
            /// </summary>
            DefinePoint = 7
        }
        #endregion

        #region Private Fields
        // Controls that will be created at startup
        /// <summary>
        /// 
        /// </summary>
        private MapControl mapControl = null;
        /// <summary>
        /// 
        /// </summary>
        private IndexMapControl indexMapControl = null;

        /// <summary>
        /// 
        /// </summary>
        private Timer _AnimateTimePointTimer = null;

        /// <summary>
        /// 
        /// </summary>
        private ModflowMetadata _Metadata = null;
        /// <summary>
        /// 
        /// </summary>
        private EndpointSymbologyDialog _EndpointSymbologyDialog = null;
        /// <summary>
        /// 
        /// </summary>
        private PathlineSymbologyDialog _PathlineSymbologyDialog = null;
        /// <summary>
        /// 
        /// </summary>
        private TimeseriesSymbologyDialog _TimeseriesSymbologyDialog = null;
        /// <summary>
        /// 
        /// </summary>
        private bool _ProcessingActiveToolButtonSelection = false;
        /// <summary>
        /// 
        /// </summary>
        private ActiveTool _ActiveTool = ActiveTool.Pointer;
        /// <summary>
        /// 
        /// </summary>
        private SimulationData _SimData = null;
        /// <summary>
        /// 
        /// </summary>
        private LayerData _HeadLayerData = null;
        /// <summary>
        /// 
        /// </summary>
        private USGS.Puma.FiniteDifference.GridGeoReference _GridGeoRef = null;
        /// <summary>
        /// 
        /// </summary>
        private USGS.Puma.FiniteDifference.CellCenteredArealGrid _ModelGrid = null;
        /// <summary>
        /// 
        /// </summary>
        private USGS.Puma.Modflow.DisFileData _DisData = null;
        /// <summary>
        /// 
        /// </summary>
        private EndpointDataset endpointDataset = new EndpointDataset();
        /// <summary>
        /// 
        /// </summary>
        private PathlineDataset pathlineDataset = new PathlineDataset();
        /// <summary>
        /// 
        /// </summary>
        private TimeseriesDataset timeseriesDataset = new TimeseriesDataset();
        /// <summary>
        /// 
        /// </summary>
        private Dictionary<int, double> _QuickViewTimeseriesValues = null;
        /// <summary>
        /// 
        /// </summary>
        private List<int> _QuickViewEndpointValues = null;
        /// <summary>
        /// 
        /// </summary>
        private List<int> _QuickViewPathlineValues = null;

        // Basemap items
        /// <summary>
        /// 
        /// </summary>
        private Basemap _Basemap = null;
        /// <summary>
        /// 
        /// </summary>
        private List<FeatureLayer> _BasemapLayers = null;
        /// <summary>
        /// 
        /// </summary>
        private bool _ShowBasemap = true;

        // Map tools and items
        /// <summary>
        /// 
        /// </summary>
        private Cursor _ReCenterCursor = null;
        /// <summary>
        /// 
        /// </summary>
        private Cursor _ZoomInCursor = null;
        /// <summary>
        /// 
        /// </summary>
        private Cursor _ZoomOutCursor = null;
        /// <summary>
        /// 
        /// </summary>
        private Feature _HotFeature = null;
        /// <summary>
        /// 
        /// </summary>
        private ToolTip _MapTip = null;
        /// <summary>
        /// 
        /// </summary>
        private FeatureLayer _GridlinesMapLayer = null;
        /// <summary>
        /// 
        /// </summary>
        private FeatureLayer _GridOutlineMapLayer = null;
        /// <summary>
        /// 
        /// </summary>
        private FeatureLayer _EndpointLayer = null;
        /// <summary>
        /// 
        /// </summary>
        private FeatureLayer _PathlineLayer = null;
        /// <summary>
        /// 
        /// </summary>
        private FeatureLayer _TimeseriesLayer = null;
        /// <summary>
        /// 
        /// </summary>
        private FeatureLayer _CurrentContourMapLayer = null;
        /// <summary>
        /// 
        /// </summary>
        private ContourEngineData _ContourEngineData = null;

        // Particle feature collections and related items
        /// <summary>
        /// 
        /// </summary>
        private FeatureCollection _PathlineFeatures = null;
        /// <summary>
        /// 
        /// </summary>
        private FeatureCollection _EndpointFeatures = null;
        /// <summary>
        /// 
        /// </summary>
        private FeatureCollection _TimeseriesFeatures = null;
        /// <summary>
        /// 
        /// </summary>
        private EndpointLocationTypes _CurrentEndpointFeatures;

        // Printer settings
        /// <summary>
        /// 
        /// </summary>
        private System.Drawing.Printing.PrinterSettings _PrinterSettings = null;
        /// <summary>
        /// 
        /// </summary>
        private System.Drawing.Printing.PageSettings _PdfPageSettings = null;

        #endregion

        #region Form Event Handlers
        /// <summary>
        /// Handles the Load event of the ModpathOutputExaminer control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void ModpathOutputExaminer_Load(object sender, EventArgs e)
        {
            SetCenteredApplicationSize(0.9, 1.5);
        }
        /// <summary>
        /// Handles the FormClosing event of the ModpathOutputExaminer control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.Windows.Forms.FormClosingEventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void ModpathOutputExaminer_FormClosing(object sender, FormClosingEventArgs e)
        {
            this.CloseDataset();
        }
        #endregion

        #region Tab Datset Event Handlers
        /// <summary>
        /// Handles the SelectedIndexChanged event of the tabDataset control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void tabDataset_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (tabDataset.SelectedTab.Tag == "map")
            {
                SetMapMenuAndToolbarAccess(true);
            }
            else
            {
                StopTimePointAnimation();
                SetMapMenuAndToolbarAccess(false);
                toolStripStatusMapXyLocation.Text = "";
                toolStripStatusMapCellLocation.Text = "";
            }
        }

        #endregion

        #region Menu Event Handlers
        /// <summary>
        /// Handles the Click event of the menuMainFile control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainFile_Click(object sender, EventArgs e)
        {
            if (this.AnimationIsRunning)
            { StopTimePointAnimation(); }
        }
        /// <summary>
        /// Handles the Click event of the menuMainEdit control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainEdit_Click(object sender, EventArgs e)
        {
            if (this.AnimationIsRunning)
            { StopTimePointAnimation(); }
        }
        /// <summary>
        /// Handles the Click event of the menuMainView control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainView_Click(object sender, EventArgs e)
        {
            if (this.AnimationIsRunning)
            { StopTimePointAnimation(); }
        }
        /// <summary>
        /// Handles the Click event of the menuMainMap control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainMap_Click(object sender, EventArgs e)
        {
            if (this.AnimationIsRunning)
            { StopTimePointAnimation(); }
        }
        /// <summary>
        /// Handles the Click event of the menuMainHelp control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainHelp_Click(object sender, EventArgs e)
        {
            if (this.AnimationIsRunning)
            { StopTimePointAnimation(); }
        }

        /// <summary>
        /// Handles the Click event of the menuMainFileOpen control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainFileOpen_Click(object sender, EventArgs e)
        {
            string filename = BrowseFile();
            if (!string.IsNullOrEmpty(filename))
            {
                OpenDataset(filename);
            }
        }
        /// <summary>
        /// Handles the Click event of the menuMainFileClose control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainFileClose_Click(object sender, EventArgs e)
        {
            CloseDataset();
        }
        /// <summary>
        /// Handles the Click event of the menuMainFileNewBasemap control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainFileNewBasemap_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Title = "New Basemap File";
            dialog.Filter = "Basemaps (*.basemap)|*.basemap|All files (*.*)|*.*";
            dialog.AddExtension = true;
            dialog.DefaultExt = "basemap";
            dialog.CheckFileExists = false;
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                if (File.Exists(dialog.FileName))
                {
                    string messageText = "The specified file already exists." + Environment.NewLine + "Select a different file name.";
                    DialogResult result = MessageBox.Show(messageText, "File exists");
                    return;
                }

                Basemap newBasemap = new Basemap();
                Basemap.Write(dialog.FileName, newBasemap);
                LoadBasemap(dialog.FileName);
                if (_Basemap != null)
                {
                    BasemapEditDialog editBasemapDialog = new BasemapEditDialog(_Basemap);
                    if (editBasemapDialog.ShowDialog() == DialogResult.OK)
                    {
                        Basemap.Write(_Basemap);
                        LoadBasemap(dialog.FileName);
                    }
                }
            }
        }
        /// <summary>
        /// Handles the Click event of the menuMainFileLoadBasemap control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainFileLoadBasemap_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Filter = "Basemap files (*.basemap)|*.basemap|All files (*.*)|*.*";
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                LoadBasemap(dialog.FileName);
            }

        }
        /// <summary>
        /// Handles the Click event of the menuMainFileRemoveBasemap control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainFileRemoveBasemap_Click(object sender, EventArgs e)
        {
            _Basemap = null;
            _BasemapLayers.Clear();
            if (_ModelGrid != null)
            { BuildMapLayers(true); }
            else
            { BuildMapLayers(false); }
        }
        /// <summary>
        /// Handles the Click event of the menuMainFileSaveBasemap control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainFileSaveBasemap_Click(object sender, EventArgs e)
        {
            if (_Basemap != null)
            {
                Basemap.Write(_Basemap);
            }
        }
        /// <summary>
        /// Handles the Click event of the menuMainFileSaveBasemapAs control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainFileSaveBasemapAs_Click(object sender, EventArgs e)
        {
            SaveBasemapAs();
        }
        /// <summary>
        /// Handles the Click event of the menuMainFileExportShapefiles control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainFileExportShapefiles_Click(object sender, EventArgs e)
        {
            if (_SimData != null)
            {

                ExportShapefilesDialog dialog = new ExportShapefilesDialog(_SimData, _ContourEngineData.SelectedDataLayer);
                dialog.GridOutlineLayer = _GridOutlineMapLayer;
                dialog.GridlinesLayer = _GridlinesMapLayer;
                dialog.ContourLayer = _CurrentContourMapLayer;
                dialog.ExportParticleLayer = true;

                switch (_SimData.SimulationType)
                {
                    case 1:
                        dialog.ParticleOutputLayer = _EndpointLayer;
                        break;
                    case 2:
                        dialog.ParticleOutputLayer = _PathlineLayer;
                        break;
                    case 3:
                        dialog.ParticleOutputLayer = _TimeseriesLayer;
                        break;
                    default:
                        break;
                }

                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    // Add code to write message to status strip
                }

            }
       }
        /// <summary>
        /// Handles the Click event of the menuMainFileImportShapefiles control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainFileImportShapefiles_Click(object sender, EventArgs e)
        {
            using (ImportShapefilesDialog dialog = new ImportShapefilesDialog())
            {
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    // nothing to do
                }
            }
        }
        /// <summary>
        /// Handles the Click event of the menuMainFileExit control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainFileExit_Click(object sender, EventArgs e)
        {
            CloseDataset();
            this.Close();
        }

        /// <summary>
        /// Handles the Click event of the menuMainEditEndpointQueryFilter control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainEditEndpointQueryFilter_Click(object sender, EventArgs e)
        {
            ShowAndProcessEndpointQueryFilter();
        }
        /// <summary>
        /// Handles the Click event of the menuMainEditMapSymbology control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainEditMapSymbology_Click(object sender, EventArgs e)
        {
            EditSymbology();
        }
        /// <summary>
        /// Handles the Click event of the menuMainEditBasemap control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainEditBasemap_Click(object sender, EventArgs e)
        {
            EditBasemap();
        }
        /// <summary>
        /// Handles the Click event of the menuMainEditMetadata control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainEditMetadata_Click(object sender, EventArgs e)
        {
            EditMetadata();
        }
        /// <summary>
        /// Handles the Click event of the menuMainEditContourData control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainEditContourData_Click(object sender, EventArgs e)
        {
            EditContourLayer();
        }

        /// <summary>
        /// Handles the Click event of the menuMainMapToolPointer control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainMapToolPointer_Click(object sender, EventArgs e)
        {
            if (!_ProcessingActiveToolButtonSelection)
            {
                SelectActiveTool(ActiveTool.Pointer);
            }

        }
        /// <summary>
        /// Handles the Click event of the menuMainMapToolReCenter control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainMapToolReCenter_Click(object sender, EventArgs e)
        {
            if (!_ProcessingActiveToolButtonSelection)
            {
                SelectActiveTool(ActiveTool.ReCenter);
            }

        }
        /// <summary>
        /// Handles the Click event of the menuMainMapToolZoomIn control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainMapToolZoomIn_Click(object sender, EventArgs e)
        {
            if (!_ProcessingActiveToolButtonSelection)
            {
                SelectActiveTool(ActiveTool.ZoomIn);
            }

        }
        /// <summary>
        /// Handles the Click event of the menuMainMapToolZoomOut control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainMapToolZoomOut_Click(object sender, EventArgs e)
        {
            if (!_ProcessingActiveToolButtonSelection)
            {
                SelectActiveTool(ActiveTool.ZoomOut);
            }

        }
        /// <summary>
        /// Handles the Click event of the menuMainMapZoomModelGrid control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainMapZoomModelGrid_Click(object sender, EventArgs e)
        {
            ZoomToGrid();

        }
        /// <summary>
        /// Handles the Click event of the menuMainMapZoomFullExtent control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainMapZoomFullExtent_Click(object sender, EventArgs e)
        {
            mapControl.SizeToFullExtent();

        }
        /// <summary>
        /// Handles the Click event of the menuMainHelpAbout control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainHelpAbout_Click(object sender, EventArgs e)
        {
            AboutBoxModpathOutputExaminer dialog = new AboutBoxModpathOutputExaminer();
            dialog.ShowDialog();
        }
        /// <summary>
        /// Handles the Click event of the menuMainFilePrintPreview control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainFilePrintPreview_Click(object sender, EventArgs e)
        {
            if (mapControl.LayerCount > 0)
            {
                PrintMap(true);
            }
            else
            {
                MessageBox.Show("The map is empty.");
            }

        }
        /// <summary>
        /// Handles the Click event of the menuMainFilePrint control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainFilePrint_Click(object sender, EventArgs e)
        {
            if (mapControl.LayerCount > 0)
            {
                PrintMap(false);
            }
            else
            {
                MessageBox.Show("The map is empty.");
            }
        }
        /// <summary>
        /// Handles the Click event of the menuMainFilePrintPDF control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void menuMainFilePrintPDF_Click(object sender, EventArgs e)
        {
            if (mapControl.LayerCount > 0)
            {
                string directoryName = null;
                if (_SimData != null)
                {
                    directoryName = _SimData.WorkingDirectory;
                }
                else
                { directoryName = @"C:"; }

                SaveFileDialog dialog = new SaveFileDialog();
                dialog.InitialDirectory = directoryName;
                dialog.FileName = "map_output.pdf";
                dialog.DefaultExt = "pdf";
                dialog.Filter = "*.pdf (Adobe PDF files)|*.pdf|*.* (All files)|*.*";
                dialog.AddExtension = true;
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    PrintPDF(dialog.FileName);
                }
            }
            else
            {
                MessageBox.Show("The map is empty.");
            }
        }

        #endregion

        #region Toolbar Event Handlers
        /// <summary>
        /// Handles the ItemClicked event of the toolStripMain control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.Windows.Forms.ToolStripItemClickedEventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripMain_ItemClicked(object sender, ToolStripItemClickedEventArgs e)
        {
            // If the start/stop animation button was clicked, just return and let things
            // happen naturally.
            if (e.ClickedItem.Name == toolStripButtonAnimateTimepoints.Name)
            { return; }

            // If any other button was clicked, stop any running animation now.
            if (this.AnimationIsRunning)
            {
                StopTimePointAnimation();
            }

        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonSelect control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonSelect_Click(object sender, EventArgs e)
        {
            if (!_ProcessingActiveToolButtonSelection)
            {
                SelectActiveTool(ActiveTool.Pointer);
            }

        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonReCenter control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonReCenter_Click(object sender, EventArgs e)
        {
            if (!_ProcessingActiveToolButtonSelection)
            {
                SelectActiveTool(ActiveTool.ReCenter);
            }

        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonZoomIn control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonZoomIn_Click(object sender, EventArgs e)
        {
            if (!_ProcessingActiveToolButtonSelection)
            {
                SelectActiveTool(ActiveTool.ZoomIn);
            }

        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonZoomOut control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonZoomOut_Click(object sender, EventArgs e)
        {
            if (!_ProcessingActiveToolButtonSelection)
            {
                SelectActiveTool(ActiveTool.ZoomOut);
            }

        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonZoomToGrid control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonZoomToGrid_Click(object sender, EventArgs e)
        {
            ZoomToGrid();

        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonFullExtent control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonFullExtent_Click(object sender, EventArgs e)
        {
            mapControl.SizeToFullExtent();

        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonEditContours control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonEditContours_Click(object sender, EventArgs e)
        {
            EditContourLayer();
        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonEditBasemap control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonEditBasemap_Click(object sender, EventArgs e)
        {
            EditBasemap();
        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonEditMetadata control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonEditMetadata_Click(object sender, EventArgs e)
        {
            EditMetadata();
        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonEditDataFilter control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonEditDataFilter_Click(object sender, EventArgs e)
        {
            if (_SimData.SimulationType == 1)
            {
                ShowAndProcessEndpointQueryFilter();
            }
            else if (_SimData.SimulationType == 2)
            {
                ShowAndProcessPathlineQueryFilter();
            }
            else if (_SimData.SimulationType == 3)
            {
                ShowAndProcessTimeseriesQueryFilter();
            }
        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonEditSymbology control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonEditSymbology_Click(object sender, EventArgs e)
        {
            EditSymbology();
        }
        /// <summary>
        /// Handles the SelectedIndexChanged event of the toolStripComboBoxQuickView control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripComboBoxQuickView_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (_SimData == null)
            { return; }

            if (_SimData.SimulationType == 1)
            {
                SelectQuickViewEndpoints(toolStripComboBoxQuickView.SelectedIndex);
            }
            else if (_SimData.SimulationType == 2)
            {
                SelectQuickViewPathlines(toolStripComboBoxQuickView.SelectedIndex);
            }
            else if (_SimData.SimulationType == 3)
            {
                SelectQuickViewTimeseries(toolStripComboBoxQuickView.SelectedIndex);
            }
        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonNextTimePoint control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonNextTimePoint_Click(object sender, EventArgs e)
        {
            int n = toolStripComboBoxQuickView.SelectedIndex + 1;
            if (n == toolStripComboBoxQuickView.Items.Count)
            {
                n = 0;
            }
            toolStripComboBoxQuickView.SelectedIndex = n;

        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonPreviousTimePoint control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonPreviousTimePoint_Click(object sender, EventArgs e)
        {
            int n = toolStripComboBoxQuickView.SelectedIndex - 1;
            if (n < 0)
            { return; }
            toolStripComboBoxQuickView.SelectedIndex = n;
        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonFirstTimePoint control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonFirstTimePoint_Click(object sender, EventArgs e)
        {
            if (toolStripComboBoxQuickView.SelectedIndex > 1)
            {
                toolStripComboBoxQuickView.SelectedIndex = 1;
            }
            else
            {
                toolStripComboBoxQuickView.SelectedIndex = 0;
            }
        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonMoveLastTimePoint control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonMoveLastTimePoint_Click(object sender, EventArgs e)
        {
            toolStripComboBoxQuickView.SelectedIndex = toolStripComboBoxQuickView.Items.Count - 1;
        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonAnimateTimepoints control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonAnimateTimepoints_Click(object sender, EventArgs e)
        {
            if (this.AnimationIsRunning)
            {
                StopTimePointAnimation();
            }
            else
            {
                StartTimePointAnimation();
            }
        }
        /// <summary>
        /// Handles the Click event of the toolStripButtonContinuousLoop control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripButtonContinuousLoop_Click(object sender, EventArgs e)
        {
            toolStripButtonContinuousLoop.Checked = !toolStripButtonContinuousLoop.Checked;
        }
        /// <summary>
        /// Handles the SelectedIndexChanged event of the toolStripComboBoxAnimationInterval control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void toolStripComboBoxAnimationInterval_SelectedIndexChanged(object sender, EventArgs e)
        {
            SetAnimationInterval(_AnimateTimePointTimer, toolStripComboBoxAnimationInterval.Text);
        }

        #endregion

        #region MapControl Event Handlers
        /// <summary>
        /// Handles the MouseMove event of the mapControl control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.Windows.Forms.MouseEventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void mapControl_MouseMove(object sender, MouseEventArgs e)
        {
            UpdateStatusBarLocationInfo(e.X, e.Y);

            switch (_ActiveTool)
            {
                case ActiveTool.Pointer:
                    FindContourLineHit(e.X, e.Y);
                    if (_HotFeature == null)
                    { mapControl.Cursor = System.Windows.Forms.Cursors.Default; }
                    else
                    { mapControl.Cursor = System.Windows.Forms.Cursors.Hand; }
                    break;
                case ActiveTool.ReCenter:
                    break;
                case ActiveTool.ZoomIn:
                    break;
                case ActiveTool.ZoomOut:
                    break;
                default:
                    break;
            }
        }
        /// <summary>
        /// Handles the MouseClick event of the mapControl control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.Windows.Forms.MouseEventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void mapControl_MouseClick(object sender, MouseEventArgs e)
        {
            if (mapControl.LayerCount > 0)
            {
                ICoordinate pt = mapControl.ToMapPoint(e.X, e.Y);
                switch (_ActiveTool)
                {
                    case ActiveTool.Pointer:
                        ShowMapTip(e.X, e.Y);
                        break;
                    case ActiveTool.ReCenter:
                        mapControl.Center = pt;
                        break;
                    case ActiveTool.ZoomIn:
                        mapControl.Zoom(2.0, pt.X, pt.Y);
                        break;
                    case ActiveTool.ZoomOut:
                        mapControl.Zoom(0.5, pt.X, pt.Y);
                        break;
                    default:
                        break;
                }
            }
        }
        /// <summary>
        /// Handles the MouseDoubleClick event of the mapControl control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.Windows.Forms.MouseEventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void mapControl_MouseDoubleClick(object sender, MouseEventArgs e)
        {
            switch (_ActiveTool)
            {
                case ActiveTool.Pointer:
                    break;
                case ActiveTool.ReCenter:
                    break;
                case ActiveTool.ZoomIn:
                    break;
                case ActiveTool.ZoomOut:
                    break;
                default:
                    break;
            }
        }

        #endregion

        #region Event Handlers
        /// <summary>
        /// Handles the Tick event of the AnimateTimePointHandler control.
        /// </summary>
        /// <param name="source">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void AnimateTimePointHandler_Tick(object source, EventArgs e)
        {
            int n = toolStripComboBoxQuickView.SelectedIndex + 1;
            if (n == toolStripComboBoxQuickView.Items.Count)
            {
                if (toolStripComboBoxQuickView.Items.Count > 1)
                {
                    toolStripComboBoxQuickView.SelectedIndex = 1;
                }
                else
                {
                    toolStripComboBoxQuickView.SelectedIndex = 0;
                }

                //if (toolStripButtonContinuousLoop.Checked)
                //{
                //    toolStripComboBoxQuickView.SelectedIndex = 1;
                //}

                //else
                //{
                //    StopTimePointAnimation();
                //    toolStripComboBoxQuickView.SelectedIndex = 0;
                //}
            }
            else
            {
                toolStripComboBoxQuickView.SelectedIndex = n;
            }

        }
        /// <summary>
        /// Handles the SelectedIndexChanged event of the listviewFiles control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void listviewFiles_SelectedIndexChanged(object sender, EventArgs e)
        {
            ListView lv = sender as ListView;
            if (lv.SelectedItems.Count > 0)
            {
                string key = lv.SelectedItems[0].Tag as string;
                SelectFilePanel(key);
            }
            else
            {
                SelectFilePanel("");
            }
        }
        /// <summary>
        /// Handles the LayerVisibilityChanged event of the legendParticles control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.EventArgs"/> instance containing the event data.</param>
        /// <remarks></remarks>
        private void legendParticles_LayerVisibilityChanged(object sender, EventArgs e)
        {
            mapControl.Refresh();
        }
        #endregion

        #region Private Methods
        /// <summary>
        /// Sets the size of the centered application.
        /// </summary>
        /// <param name="widthPercent">The width percent.</param>
        /// <param name="heightPercent">The height percent.</param>
        /// <remarks></remarks>
        private void SetCenteredApplicationSize(double heightPercent, double aspect)
        {
            Screen scr = Screen.PrimaryScreen;
            double h = heightPercent * Convert.ToDouble(scr.WorkingArea.Height);
            this.Height = Convert.ToInt32(h);
            this.Width = Convert.ToInt32(aspect * h);
            this.Left = (scr.WorkingArea.Width - this.Width) / 2;
            this.Top = (scr.WorkingArea.Height - this.Height) / 2;
        }
        /// <summary>
        /// Edits the basemap.
        /// </summary>
        /// <remarks></remarks>
        private void EditBasemap()
        {
            if (_Basemap != null)
            {
                BasemapEditDialog dialog = new BasemapEditDialog(_Basemap);
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    _BasemapLayers = _Basemap.CreateBasemapLayers();
                    BuildMapLayers(false);
                    indexMapControl.UpdateMapImage();
                }
            }
        }
        /// <summary>
        /// Builds the map layers.
        /// </summary>
        /// <param name="fullExtent">if set to <c>true</c> [full extent].</param>
        /// <remarks></remarks>
        private void BuildMapLayers(bool fullExtent)
        {
            bool forceFullExtent = fullExtent;
            if (mapControl.LayerCount == 0)
                forceFullExtent = true;

            mapControl.ClearLayers();

            // Grid outline
            if (_GridOutlineMapLayer != null)
            { mapControl.AddLayer(_GridOutlineMapLayer); }

            // Basemap layers (add in reverse order)
            if (_ShowBasemap && (_Basemap != null))
            {
                if (_BasemapLayers.Count > 0)
                {
                    for (int i = _BasemapLayers.Count - 1; i > -1; i--)
                    {
                        mapControl.AddLayer(_BasemapLayers[i]);
                    }
                }
            }


            // Endpoint layer
            if (_SimData.SimulationType == 1)
            {
                if (_EndpointLayer == null)
                {
                    if (_EndpointFeatures == null)
                    {
                        _CurrentEndpointFeatures = _EndpointSymbologyDialog.EndpointOption;
                        _EndpointFeatures = ParticleFeatures.CreateEndpointFeatures(endpointDataset.FilteredRecords, _CurrentEndpointFeatures);
                        // Transform to global coordinates, if necessary
                        TransformFeatures(_EndpointFeatures, _ModelGrid, true);

                        if (_SimData.TrackingDirection == 1)
                        {
                            _QuickViewEndpointValues = endpointDataset.GetAvailableFinalZones();
                        }
                        else
                        {
                            _QuickViewEndpointValues = endpointDataset.GetAvailableInitialZones();
                        }
                    }

                    if (_SimData.TrackingDirection == 1)
                    {
                        InitializeQuickViewEndpoints(_QuickViewEndpointValues, "FinalZone");
                    }
                    else
                    {
                        InitializeQuickViewEndpoints(_QuickViewEndpointValues, "InitZone");
                    }

                    _EndpointLayer = CreateEndpointsLayer(_EndpointFeatures);
                    if (_CurrentEndpointFeatures == EndpointLocationTypes.InitialPoint)
                    {
                        _EndpointLayer.LayerName = "Initial locations";
                    }
                    else if (_CurrentEndpointFeatures == EndpointLocationTypes.FinalPoint)
                    {
                        _EndpointLayer.LayerName = "Final locations";
                    }

                    
                }

                SelectQuickViewEndpoints(toolStripComboBoxQuickView.SelectedIndex);

            }


            if (_EndpointLayer != null)
            {
                mapControl.AddLayer(_EndpointLayer);
            }

            // Timeseries layer
            if (_SimData.SimulationType == 3)
            {
                if (_TimeseriesLayer == null)
                {
                    if (_TimeseriesFeatures == null)
                    {
                        _TimeseriesFeatures = ParticleFeatures.CreateTimeseriesFeatures(timeseriesDataset.FilteredRecords, endpointDataset.TotalRecords, false);
                        _QuickViewTimeseriesValues = timeseriesDataset.GetAvailableTimePoints();
                        // Transform to global coordinates, if necessary
                        TransformFeatures(_TimeseriesFeatures, _ModelGrid, true);

                    }

                    InitializeQuickViewTimeseries(_QuickViewTimeseriesValues);

                    _TimeseriesLayer = CreateTimeseriesLayer(_TimeseriesFeatures);
                    SelectQuickViewTimeseries(0);

                }
            }

            if (_TimeseriesLayer != null)
            {
                mapControl.AddLayer(_TimeseriesLayer);
            }

            

            // Interior grid lines
            if (_GridlinesMapLayer != null)
            { mapControl.AddLayer(_GridlinesMapLayer); }

            //// Basemap layers (add in reverse order)
            //if (_ShowBasemap && (_Basemap != null))
            //{
            //    if (_BasemapLayers.Count > 0)
            //    {
            //        for (int i = _BasemapLayers.Count - 1; i > -1; i--)
            //        {
            //            mapControl.AddLayer(_BasemapLayers[i]);
            //        }
            //    }
            //}

            // Head contours
            if (_CurrentContourMapLayer != null)
            {
                mapControl.AddLayer(_CurrentContourMapLayer);
            }

            // Pathline layer
            if (_SimData.SimulationType == 2)
            {
                if (_PathlineLayer == null)
                {
                    if (_PathlineFeatures == null)
                    {
                        _PathlineFeatures = ParticleFeatures.CreatePathlineFeatures(pathlineDataset.FilteredRecords, endpointDataset.TotalRecords, false);
                        // Transform to global coordinates, if necessary
                        TransformFeatures(_PathlineFeatures, _ModelGrid, true);

                        if (_SimData.TrackingDirection == 1)
                        {
                            _QuickViewPathlineValues = endpointDataset.GetAvailableFinalZones();
                        }
                        else
                        {
                            _QuickViewPathlineValues = endpointDataset.GetAvailableInitialZones();
                        }
                    }

                    if (_SimData.TrackingDirection == 1)
                    {
                        InitializeQuickViewPathlines(_QuickViewPathlineValues, "FinalZone");
                    }
                    else
                    {
                        InitializeQuickViewPathlines(_QuickViewPathlineValues, "InitZone");
                    }

                    _PathlineLayer = CreatePathlinesLayer(_PathlineFeatures);
                }

                SelectQuickViewPathlines(toolStripComboBoxQuickView.SelectedIndex);

                if(_PathlineLayer!=null)
                {
                    mapControl.AddLayer(_PathlineLayer);
                }
            }

            // Prepare and display the map
            if (mapControl.LayerCount > 0)
            {
                if (forceFullExtent)
                {
                    if (_ModelGrid == null)
                    { mapControl.SizeToFullExtent(); }
                    else
                    { ZoomToGrid(); }
                }
            }
            BuildMapLegend();
            mapControl.Refresh();
            indexMapControl.UpdateMapImage();

        }
        /// <summary>
        /// Browses the file.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        private string BrowseFile()
        {
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Filter = "Modpath Simulation files (*.mpsim)|*.mpsim|All files (*.*)|*.*";
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                this.Refresh();
                return dialog.FileName;
            }
            return "";
        }
        /// <summary>
        /// Opens the dataset.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <remarks></remarks>
        private void OpenDataset(string filename)
        {
            bool viewParticleTextFiles = false;

            if (filename != "")
            {
                string pathname = "";
                string endpointFilename = "";
                string pathlineFilename = "";
                string timeseriesFilename = "";

                // Close any existing dataset
                CloseDataset();
                this.Refresh();

                
                List<string> fileLines = new List<string>();

                listviewFiles.Items.Clear();
                string[] subitems = new string[3];
                ListViewItem lvItem;

                toolStripStatusMessage.Text = "Loading simulation data ...";
                statusStripMain.Refresh();

                // Read simulation file
                SimulationFileReader reader = new SimulationFileReader(filename);
                _SimData = reader.Read();
                
                // Initialize endpoint symbology
                _EndpointSymbologyDialog = new EndpointSymbologyDialog();
                _EndpointSymbologyDialog.RenderOption = EndpointRenderOptionTypes.UniqueValues;
                if (_SimData.TrackingDirection == 1)
                {
                    _EndpointSymbologyDialog.UniqueValuesRenderField = "FinalZone";
                    _EndpointSymbologyDialog.EndpointOption = EndpointLocationTypes.InitialPoint;
                }
                else
                {
                    _EndpointSymbologyDialog.UniqueValuesRenderField = "InitZone";
                    _EndpointSymbologyDialog.EndpointOption = EndpointLocationTypes.FinalPoint;
                }
                _EndpointSymbologyDialog.SymbolType = PointSymbolTypes.Square;

                // Read and process endpoint file
                endpointFilename = _SimData.GetFullPathname("endpoint");
                endpointDataset.LoadFromFile(endpointFilename);
                textboxDataFilterSummary.Text = endpointDataset.QueryFilter.Summary;
                if (viewParticleTextFiles)
                {
                    fileLines.Clear();
                    ReadTextFile(endpointFilename, fileLines);
                    textboxEndpointsFile.Lines = fileLines.ToArray();
                }

                // Read and process pathline file
                if (_SimData.SimulationType == 2)
                {
                    pathlineFilename = _SimData.GetFullPathname("pathline");
                    pathlineDataset.LoadFromFile(pathlineFilename);
                    if (viewParticleTextFiles)
                    {
                        fileLines.Clear();
                        ReadTextFile(pathlineFilename, fileLines);
                        textboxPathlinesFile.Lines = fileLines.ToArray();
                    }
                    _PathlineSymbologyDialog = new PathlineSymbologyDialog();
                    _PathlineSymbologyDialog.LineWidth = 2.0f;
                    _PathlineSymbologyDialog.RenderOption = PathlineRenderOptionTypes.SingleSymbol;
                    _PathlineSymbologyDialog.UniqueValuesRenderField = "Group";
                }
                
                // Read and process timeseries file
                if (_SimData.SimulationType == 3)
                {
                    timeseriesFilename = _SimData.GetFullPathname("timeseries");
                    timeseriesDataset.LoadFromFile(timeseriesFilename);
                    if (viewParticleTextFiles)
                    {
                        fileLines.Clear();
                        ReadTextFile(timeseriesFilename, fileLines);
                        textboxTimeseriesFile.Lines = fileLines.ToArray();
                    }
                    _TimeseriesSymbologyDialog = new TimeseriesSymbologyDialog();
                    _TimeseriesSymbologyDialog.SymbolSize = 2.0f;
                    _TimeseriesSymbologyDialog.SymbolType = PointSymbolTypes.Square;
                    _TimeseriesSymbologyDialog.RenderOption = TimeseriesRenderOptionTypes.SingleSymbol;
                    _TimeseriesSymbologyDialog.ContinuousValuesRenderField = "Time";
                    _TimeseriesSymbologyDialog.UniqueValuesRenderField = "TimePoint";

                    EnableTimePointNavigationControls(true, true);
                }

                // Process MPSIM file
                fileLines.Clear();
                ReadTextFile(filename, fileLines);
                textboxMPSIMfile.Clear();
                textboxMPSIMfile.Lines = fileLines.ToArray<string>();
                //textboxSummary.Text = "\n" + "Simulation file: " + filename;
                subitems[0] = "mpsim";
                subitems[1] = "Modpath simulation file";
                subitems[2] = filename;
                lvItem = new ListViewItem(subitems);
                lvItem.Tag = "mpsim";
                listviewFiles.Items.Add(lvItem);

                // Process MPNAM file
                pathname = _SimData.GetFullPathname("mpnam");
                fileLines.Clear();
                ReadTextFile(pathname, fileLines);
                textboxMPNAMfile.Clear();
                textboxMPNAMfile.Lines = fileLines.ToArray<string>();
                List<ModpathNameFileItem> nameFileItems = ReadModpathNameFile(pathname);
                subitems[0] = "mpnam";
                subitems[1] = "Modpath name file";
                subitems[2] = Path.Combine(_SimData.WorkingDirectory, _SimData.NameFile);
                lvItem = new ListViewItem(subitems);
                lvItem.Tag = "mpnam";
                listviewFiles.Items.Add(lvItem);

                // Process DIS file
                foreach (ModpathNameFileItem item in nameFileItems)
                {
                    string fileType = item.FileType;
                    if (item.FileType == "dis")
                    {
                        pathname = item.Filename;
                        break;
                    }
                }
                pathname = _SimData.ConvertToFullPathname(pathname);
                fileLines.Clear();
                ReadTextFile(pathname, fileLines);
                textboxDISfile.Clear();
                textboxDISfile.Lines = fileLines.ToArray<string>();
                subitems[0] = "dis";
                subitems[1] = "Modflow discretization file";
                subitems[2] = pathname;
                lvItem = new ListViewItem(subitems);
                lvItem.Tag = "dis";
                listviewFiles.Items.Add(lvItem);
                
                // Read a ModflowMetadata file if one exists. Otherwise, create 
                // the file now.
                string metadataFile = pathname + ".metadata";
                if(File.Exists(metadataFile))
                {
                    _Metadata = ModflowMetadata.Read(metadataFile);
                    _Metadata.BasemapFile = _Metadata.BasemapFile.Trim();
                    if (!string.IsNullOrEmpty(_Metadata.BasemapFile))
                    {
                        if (!Path.IsPathRooted(_Metadata.BasemapFile))
                        {
                            _Metadata.BasemapFile = Path.Combine(_Metadata.SourcefileDirectory, _Metadata.BasemapFile);
                        }
                    }
                }
                else
                {
                    _Metadata = new ModflowMetadata();
                    _Metadata.SourcefileDirectory = Path.GetDirectoryName(metadataFile);
                    _Metadata.Filename = metadataFile;
                    ModflowMetadata.Write(metadataFile, _Metadata);
                }

                // Create Model Grid and the Model Grid outline and gridlines 
                // feature layers.
                CreateModelGrid(pathname);
                GridGeoReference geoRef = _Metadata.GridGeoReference;
                _ModelGrid.OriginX = geoRef.OriginX;
                _ModelGrid.OriginY = geoRef.OriginY;
                _ModelGrid.Angle = geoRef.Angle;
                _GridlinesMapLayer = CreateModelGridlinesLayer(_ModelGrid, Color.DarkGray);
                _GridOutlineMapLayer = CreateModelGridOutlineLayer(_ModelGrid, Color.Black);

                // Process MPBAS file
                foreach (ModpathNameFileItem item in nameFileItems)
                {
                    if (item.FileType == "mpbas")
                    {
                        pathname = item.Filename;
                        break;
                    }
                }
                pathname = _SimData.ConvertToFullPathname(pathname);
                fileLines.Clear();
                ReadTextFile(pathname, fileLines);
                textboxMPBASfile.Clear();
                textboxMPBASfile.Lines = fileLines.ToArray<string>();
                subitems[0] = "mpbas";
                subitems[1] = "Modpath basic input file";
                subitems[2] = pathname;
                lvItem = new ListViewItem(subitems);
                lvItem.Tag = "mpbas";
                listviewFiles.Items.Add(lvItem);
                float hnflo = 1.0E+30f;
                float hdry = 1.0E+20f;
                foreach (string line in fileLines)
                {
                    if (line[0] != '#')
                    {
                        List<string> tokens = StringUtility.ParseAsFortranFreeFormat(line, false);
                        if (tokens.Count > 1)
                        {
                            hnflo = float.Parse(tokens[0]);
                            hdry = float.Parse(tokens[1]);
                            break;
                        }
                    }
                }

                // Process HEAD file
                string headFilename = "";
                foreach (ModpathNameFileItem item in nameFileItems)
                {
                    string fileType = item.FileType;
                    if (item.FileType == "head")
                    {
                        headFilename = item.Filename;
                        break;
                    }
                }
                if (!string.IsNullOrEmpty(headFilename))
                {
                    headFilename = _SimData.ConvertToFullPathname(headFilename);
                }
                else
                {
                    _HeadLayerData = null;
                }

                // Set contour data
                if (_ContourEngineData == null)
                {
                    _ContourEngineData = new ContourEngineData();
                }
                // Set binary reader
                BinaryLayerReader headReader = new BinaryLayerReader(headFilename);
                _ContourEngineData.ContourSourceFile = headReader;
                _ContourEngineData.SelectedDataLayer = headReader.ReadRecordHeader(0);
                // Excluded values
                _ContourEngineData.ExcludedValues.Clear();
                _ContourEngineData.ExcludedValues.Add(hnflo);
                if (!_ContourEngineData.ExcludedValues.Contains(hdry))
                {
                    _ContourEngineData.ExcludedValues.Add(hdry);
                }



                // Process MPLIST file
                pathname = _SimData.GetFullPathname("mplist");
                fileLines.Clear();
                ReadTextFile(pathname, fileLines);
                textboxMPLISTfile.Clear();
                textboxMPLISTfile.Lines = fileLines.ToArray<string>();
                subitems[0] = "mplist";
                subitems[1] = "Modpath listing file";
                subitems[2] = pathname;
                lvItem = new ListViewItem(subitems);
                lvItem.Tag = "mplist";
                listviewFiles.Items.Add(lvItem);

                // Process TRACE file
                pathname = _SimData.GetFullPathname("trace");
                if (!string.IsNullOrEmpty(pathname))
                {
                    fileLines.Clear();
                    ReadTextFile(pathname, fileLines);
                    textboxTraceFile.Clear();
                    textboxTraceFile.Lines = fileLines.ToArray<string>();
                    subitems[0] = "trace";
                    subitems[1] = "Modpath trace file";
                    subitems[2] = pathname;
                    lvItem = new ListViewItem(subitems);
                    lvItem.Tag = "trace";
                    listviewFiles.Items.Add(lvItem);
                }

                // Add endpoints file to the files listview
                if (viewParticleTextFiles)
                {
                    subitems[0] = "endpoints";
                    subitems[1] = "Modpath endpoint file";
                    subitems[2] = endpointFilename;
                    lvItem = new ListViewItem(subitems);
                    lvItem.Tag = "endpoints";
                    listviewFiles.Items.Add(lvItem);
                }

                // Add pathline file to the files listview
                if ((_SimData.SimulationType == 2) && viewParticleTextFiles)
                {
                    subitems[0] = "pathlines";
                    subitems[1] = "Modpath pathline file";
                    subitems[2] = pathlineFilename;
                    lvItem = new ListViewItem(subitems);
                    lvItem.Tag = "pathlines";
                    listviewFiles.Items.Add(lvItem);
                }

                // Add timeseries file to the files listview
                if ((_SimData.SimulationType == 3) && viewParticleTextFiles)
                {
                    subitems[0] = "timeseries";
                    subitems[1] = "Modpath timeseries file";
                    subitems[2] = timeseriesFilename;
                    lvItem = new ListViewItem(subitems);
                    lvItem.Tag = "timeseries";
                    listviewFiles.Items.Add(lvItem);
                }
                
                // Add tab pages
                //tabDataset.TabPages.Add(tabPageFiles);
                //tabDataset.TabPages.Add(tabPageMap);

                // Select simulation file
                listviewFiles.SelectedIndices.Add(0);

                tabDataset.Enabled = true;
                tabMapInfo.SelectTab(tabPageMapInfoLegend);
                SetMapMenuAndToolbarAccess(true);

                statusStripMain.Items[0].Text = "";

                // Select the Map panel tab
                tabDataset.SelectTab(tabPageMap);
                // Make sure the map toolbar is enabled
                toolStripMain.Enabled = true;

                // Generate contours
                LayerDataRecord<float> headLayerRecord = _ContourEngineData.ContourSourceFile.GetRecordAsSingle(_ContourEngineData.SelectedDataLayer);
                if (headLayerRecord != null)
                {
                    GenerateAndBuildContourLayer(headLayerRecord.DataArray, _ModelGrid);
                }
                else
                {
                    _CurrentContourMapLayer = null;
                }

                // Update title bar with dataset name
                if (_SimData != null)
                {
                    this.Text = _SimData.SimulationName + " - Modpath Output Examiner";
                }
                else
                {
                    this.Text = "Modpath Output Examiner";
                }

                // Build map
                // The LoadBasemap method calls BuildMapLayers when it finishes.
                // If no basemap is loaded, call BuildMapLayers directly.
                if (!string.IsNullOrEmpty(_Metadata.BasemapFile))
                {
                    LoadBasemap(_Metadata.BasemapFile);
                }
                else
                {
                    BuildMapLayers(false);
                }
                BuildMapLegend();


            }
        }
        /// <summary>
        /// Closes the dataset.
        /// </summary>
        /// <remarks></remarks>
        private void CloseDataset()
        {
            endpointDataset.CloseDataset();
            pathlineDataset.CloseDataset();
            timeseriesDataset.CloseDataset();
            tabDataset.Enabled = false;
            toolStripMain.Enabled = false;
            toolStripStatusMapXyLocation.Text = "";
            toolStripStatusMapCellLocation.Text = "";
            toolStripStatusMessage.Text = "No dataset is loaded.";
            textboxMPSIMfile.Clear();
            textboxMPLISTfile.Clear();
            ClearMapLegend();
            _ModelGrid = null;
            _DisData = null;
            _EndpointFeatures = null;
            _EndpointLayer = null;
            _PathlineFeatures = null;
            _PathlineLayer = null;
            _TimeseriesFeatures = null;
            _TimeseriesLayer = null;
            mapControl.ClearLayers();
            legendParticles.Clear();
            legendParticles.LegendTitle = "";
            textboxDataFilterSummary.Text = "";
            toolStripComboBoxQuickView.Items.Clear();
            EnableTimePointNavigationControls(false, false);
            SelectFilePanel("");
            _SimData = null;
            if (_HeadLayerData != null)
            {
                _HeadLayerData.CloseFile();
            }
            _HeadLayerData = null;
            tabDataset.SelectTab(tabPageMap);
            tabMapInfo.SelectTab(tabPageMapInfoLegend);
            tabDataset.Refresh();
            this.Text = "Modpath Output Examiner";
        }
        /// <summary>
        /// Reads the text file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="lines">The lines.</param>
        /// <remarks></remarks>
        private void ReadTextFile(string filename, List<string> lines)
        {
            using (StreamReader reader = new StreamReader(filename))
            {
                lines.Clear();
                string line;
                while (!reader.EndOfStream)
                {
                    line = reader.ReadLine().Trim();
                    line = line.Replace('\0', ' ');
                    lines.Add(line);
                }
            }
        }
        /// <summary>
        /// Selects the file panel.
        /// </summary>
        /// <param name="key">The key.</param>
        /// <remarks></remarks>
        private void SelectFilePanel(string key)
        {
            // Turn of all file panels
            foreach (Control c in splitConFiles.Panel2.Controls)
            {
                if (c.Dock != DockStyle.Fill)
                { c.Dock = DockStyle.Fill; }

                if (c.Tag == key)
                {
                    c.BringToFront();
                    return;
                }
            }

            textbonFilesNone.BringToFront();

        }
        /// <summary>
        /// Reads the modpath name file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private List<ModpathNameFileItem> ReadModpathNameFile(string filename)
        {
            List<ModpathNameFileItem> list = new List<ModpathNameFileItem>();
            string line;
            string[] tokens;
            char[] delimiter = new char[1] { ' ' };

            using (StreamReader reader = new StreamReader(filename))
            {
                while (!reader.EndOfStream)
                {
                    line = reader.ReadLine().Trim();
                    if (!string.IsNullOrEmpty(line))
                    {
                        if (line[0] != '#')
                        {
                            line = line.Replace(',', ' ');
                            tokens = line.Split(delimiter, StringSplitOptions.RemoveEmptyEntries);
                            ModpathNameFileItem fileItem = new ModpathNameFileItem();
                            fileItem.FileType = tokens[0].ToLower();
                            fileItem.FileUnit = int.Parse(tokens[1]);
                            fileItem.Filename = tokens[2];
                            list.Add(fileItem);
                        }
                    }
                }
            }

            return list;

        }
        /// <summary>
        /// Creates the model grid.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <remarks></remarks>
        private void CreateModelGrid(string filename)
        {
            // Process DIS file and extract the areal grid.
            // The DIS file cannot contain external arrays connected by
            // unit number because no Modflow name file is provided to
            // supply unit number connections.
            DisDataReader disReader = new DisDataReader();
            //DisFileData disData = disReader.Read(filename, "dis");
            _DisData = disReader.Read(filename, "dis");
            _ModelGrid = _DisData.CreateCellCenteredGrid();

        }
        /// <summary>
        /// Selects the active tool.
        /// </summary>
        /// <param name="tool">The tool.</param>
        /// <remarks></remarks>
        private void SelectActiveTool(ActiveTool tool)
        {
            // Process the selection
            _ProcessingActiveToolButtonSelection = true;

            toolStripButtonSelect.Checked = false;
            menuMainMapToolPointer.Checked = false;
            toolStripButtonReCenter.Checked = false;
            menuMainMapToolReCenter.Checked = false;
            toolStripButtonZoomIn.Checked = false;
            menuMainMapToolZoomIn.Checked = false;
            toolStripButtonZoomOut.Checked = false;
            menuMainMapToolZoomOut.Checked = false;

            switch (tool)
            {
                case ActiveTool.Pointer:
                    _ActiveTool = tool;
                    mapControl.Cursor = System.Windows.Forms.Cursors.Default;
                    toolStripButtonSelect.Checked = true;
                    menuMainMapToolPointer.Checked = true;
                    break;
                case ActiveTool.ReCenter:
                    _ActiveTool = tool;
                    mapControl.Cursor = _ReCenterCursor;
                    toolStripButtonReCenter.Checked = true;
                    menuMainMapToolReCenter.Checked = true;
                    break;
                case ActiveTool.ZoomIn:
                    _ActiveTool = tool;
                    mapControl.Cursor = _ZoomInCursor;
                    toolStripButtonZoomIn.Checked = true;
                    menuMainMapToolZoomIn.Checked = true;
                    break;
                case ActiveTool.ZoomOut:
                    _ActiveTool = tool;
                    mapControl.Cursor = _ZoomOutCursor;
                    toolStripButtonZoomOut.Checked = true;
                    menuMainMapToolZoomOut.Checked = true;
                    break;
                default:
                    throw new ArgumentException();
            }

            _ProcessingActiveToolButtonSelection = false;

        }
        /// <summary>
        /// Zooms to grid.
        /// </summary>
        /// <remarks></remarks>
        private void ZoomToGrid()
        {
            if (_GridOutlineMapLayer != null)
            {
                IEnvelope rect = _GridOutlineMapLayer.Extent;
                mapControl.SetExtent(rect.MinX, rect.MaxX, rect.MinY, rect.MaxY);
            }
        }
        /// <summary>
        /// Sets the map menu and toolbar access.
        /// </summary>
        /// <param name="enabled">if set to <c>true</c> [enabled].</param>
        /// <remarks></remarks>
        private void SetMapMenuAndToolbarAccess(bool enabled)
        {
            // Enable/Disable the menu options related to map view.
            menuMainEditBasemap.Enabled = enabled;
            menuMainEditEndpointQueryFilter.Enabled = enabled;
            menuMainEditMapSymbology.Enabled = enabled;
            menuMainMapTool.Enabled = enabled;
            menuMainMapZoomFullExtent.Enabled = enabled;
            menuMainMapZoomModelGrid.Enabled = enabled;
            menuMainEditContourData.Enabled = enabled;
            menuMainEditMetadata.Enabled = enabled;

            // Enable/Disable everything on the toolbar because it all
            // relates to the map view.
            toolStripMain.Enabled = enabled;
        }
        /// <summary>
        /// Creates the model grid outline layer.
        /// </summary>
        /// <param name="modelGrid">The model grid.</param>
        /// <param name="color">The color.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private FeatureLayer CreateModelGridOutlineLayer(CellCenteredArealGrid modelGrid, Color color)
        {
            Array2d<float> bottomElevation = null;
            Array2d<float> topElevation = null;
            if (_ModelGrid == null)
                return null;

            if (_DisData != null)
            {
                bottomElevation = _DisData.GetBottom(_DisData.LayerCount).GetDataArrayCopy(true);
                topElevation = _DisData.Top.GetDataArrayCopy(true);
            }
            else
            {
                bottomElevation = new Array2d<float>(modelGrid.RowCount, modelGrid.ColumnCount, 0.0f);
                topElevation = new Array2d<float>(modelGrid.RowCount, modelGrid.ColumnCount, 0.0f);
            }
            IMultiLineString outline = modelGrid.GetOutline(new GridCell(1, 1), new GridCell(modelGrid.RowCount, modelGrid.ColumnCount), topElevation, bottomElevation);
            if (outline == null)
                throw new Exception("The model grid outline was not created.");

            FeatureLayer layer = new FeatureLayer(LayerGeometryType.Line);
            ILineSymbol symbol = ((ISingleSymbolRenderer)(layer.Renderer)).Symbol as ILineSymbol;
            symbol.Color = color;
            symbol.Width = 1.0f;

            IAttributesTable attributes = new AttributesTable();
            int value = 0;
            attributes.AddAttribute("Value", value);
            layer.AddFeature(outline as IGeometry, attributes);
            layer.LayerName = "Model grid outline";
            return layer;

        }
        /// <summary>
        /// Creates the pathlines layer.
        /// </summary>
        /// <param name="pathlines">The pathlines.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private FeatureLayer CreatePathlinesLayer(FeatureCollection pathlines)
        {
            return CreatePathlinesLayer(pathlines, null);
        }
        /// <summary>
        /// Creates the pathlines layer.
        /// </summary>
        /// <param name="pathlines">The pathlines.</param>
        /// <param name="lineSymbol">The line symbol.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private FeatureLayer CreatePathlinesLayer(FeatureCollection pathlines, ILineSymbol lineSymbol)
        {
            if (pathlines == null)
            { return null; }

            FeatureLayer layer = new FeatureLayer(LayerGeometryType.Line);

            if (pathlines.Count > 0)
            {
                for (int i = 0; i < pathlines.Count; i++)
                {
                    layer.AddFeature(pathlines[i]);
                }
            }

            if (_PathlineSymbologyDialog != null)
            {
                _PathlineSymbologyDialog.SetRenderer(layer);
            }

            layer.LayerName = "Pathlines";
            return layer;

        }
        /// <summary>
        /// Creates the endpoints layer.
        /// </summary>
        /// <param name="endpoints">The endpoints.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private FeatureLayer CreateEndpointsLayer(FeatureCollection endpoints)
        {
            if (endpoints == null)
            { return null; }

            FeatureLayer layer = new FeatureLayer(LayerGeometryType.Point);

            if (endpoints.Count > 0)
            {
                for (int i = 0; i < endpoints.Count; i++)
                {
                    layer.AddFeature(endpoints[i]);
                }
            }

            if (_EndpointSymbologyDialog != null)
            {
                _EndpointSymbologyDialog.SetRenderer(layer);
            }

            layer.LayerName = "Endpoints";
            return layer;

        }
        /// <summary>
        /// Creates the timeseries layer.
        /// </summary>
        /// <param name="timeseries">The timeseries.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private FeatureLayer CreateTimeseriesLayer(FeatureCollection timeseries)
        {
            if (timeseries == null)
            { return null; }

            FeatureLayer layer = new FeatureLayer(LayerGeometryType.Point);

            if (timeseries.Count > 0)
            {
                for (int i = 0; i < timeseries.Count; i++)
                {
                    layer.AddFeature(timeseries[i]);
                }
            }

            if (_TimeseriesSymbologyDialog != null)
            {
                _TimeseriesSymbologyDialog.SetRenderer(layer);
            }

            layer.LayerName = "Timeseries";
            return layer;

        }
        /// <summary>
        /// Creates the model gridlines layer.
        /// </summary>
        /// <param name="modelGrid">The model grid.</param>
        /// <param name="color">The color.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private FeatureLayer CreateModelGridlinesLayer(CellCenteredArealGrid modelGrid, Color color)
        {
            IMultiLineString outline = modelGrid.GetOutline();
            IMultiLineString[] gridlines = modelGrid.GetGridLines();

            FeatureLayer layer = new FeatureLayer(LayerGeometryType.Line);
            ILineSymbol symbol = ((ISingleSymbolRenderer)(layer.Renderer)).Symbol as ILineSymbol;
            symbol.Color = color;
            symbol.Width = 1.0f;
            
            IAttributesTable attributes = null;
            int value = 1;
            for (int i = 0; i < gridlines.Length; i++)
            {
                attributes = new AttributesTable();
                attributes.AddAttribute("Value", value);
                layer.AddFeature(gridlines[i] as IGeometry, attributes);
            }

            attributes = new AttributesTable();
            value = 0;
            attributes.AddAttribute("Value", value);
            layer.AddFeature(outline as IGeometry, attributes);
            layer.Visible = false;

            layer.LayerName = "Model gridlines";
            return layer;

        }
        /// <summary>
        /// Creates the model grid cell polygons layer.
        /// </summary>
        /// <param name="modelGrid">The model grid.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private FeatureLayer CreateModelGridCellPolygonsLayer(CellCenteredArealGrid modelGrid)
        {
            if (modelGrid == null)
            { return null; }

            Dictionary<string, Array2d<float>> dict = new Dictionary<string, Array2d<float>>();
            dict.Add("Value", new Array2d<float>(modelGrid.RowCount, modelGrid.ColumnCount));
            Feature[] features = USGS.Puma.Utilities.GeometryFactory.CreateGridCellPolygons((ICellCenteredArealGrid)modelGrid, dict);
            FeatureLayer layer = new FeatureLayer(LayerGeometryType.Polygon, features);
            layer.LayerName = "Model grid cells";
            return layer;

        }
        /// <summary>
        /// Creates the unique value point renderer.
        /// </summary>
        /// <param name="features">The features.</param>
        /// <param name="renderField">The render field.</param>
        /// <param name="symbolTemplate">The symbol template.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private IFeatureRenderer CreateUniqueValuePointRenderer(FeatureCollection features, string renderField, IPointSymbol symbolTemplate)
        {
            double[] zones = new double[features.Count];
            for (int i = 0; i < features.Count; i++)
            {
                zones[i] = Convert.ToDouble(features[i].Attributes["FinalZone"]);
            }
            IArrayUtility<double> autil = new ArrayUtility();
            double[] uvalues = autil.GetUniqueValues(zones).ToArray<double>();
            NumericValueRenderer uvRenderer = new NumericValueRenderer(SymbolType.PointSymbol, uvalues);
            IPointSymbol pointSymbol;
            for (int i = 0; i < uvRenderer.ValueCount; i++)
            {
                pointSymbol = uvRenderer.Symbols[i] as IPointSymbol;
                pointSymbol.Size = symbolTemplate.Size;
                pointSymbol.IsFilled = symbolTemplate.IsFilled;
                pointSymbol.EnableOutline = symbolTemplate.EnableOutline;
            }
            uvRenderer.RenderField = renderField;

            return uvRenderer as IFeatureRenderer;

        }
        /// <summary>
        /// Shows the and process endpoint query filter.
        /// </summary>
        /// <remarks></remarks>
        private void ShowAndProcessEndpointQueryFilter()
        {
            if (endpointDataset != null)
            {
                if (endpointDataset.TotalRecords.Count > 0)
                {
                    endpointDataset.QueryFilter.DataSource = endpointDataset.TotalRecords;
                    if (endpointDataset.QueryFilter.ShowFilterDialog())
                    {
                        List<EndpointRecord> list = endpointDataset.QueryFilter.Execute();
                        if (list == null)
                        { list = new List<EndpointRecord>(); }
                        endpointDataset.FilteredRecords = list;
                        textboxDataFilterSummary.Text = endpointDataset.QueryFilter.Summary;
                        _EndpointFeatures = null;
                        _EndpointLayer = null;
                        BuildMapLayers(false);
                    }
                    endpointDataset.QueryFilter.DataSource = null;
                }
            }

        }
        /// <summary>
        /// Shows the and process pathline query filter.
        /// </summary>
        /// <remarks></remarks>
        private void ShowAndProcessPathlineQueryFilter()
        {
            if (pathlineDataset != null)
            {
                if (pathlineDataset.TotalRecords.Count > 0)
                {
                    pathlineDataset.QueryFilter.DataSource = pathlineDataset.TotalRecords;
                    if (pathlineDataset.QueryFilter.ShowFilterDialog())
                    {
                        List<PathlineRecord> list = pathlineDataset.QueryFilter.Execute();
                        if (list == null)
                        { list = new List<PathlineRecord>(); }
                        pathlineDataset.FilteredRecords = list;
                        textboxDataFilterSummary.Text = pathlineDataset.QueryFilter.Summary;
                        _PathlineFeatures = null;
                        _PathlineLayer = null;
                        BuildMapLayers(false);
                    }
                    pathlineDataset.QueryFilter.DataSource = null;
                }
            }


        }
        /// <summary>
        /// Shows the and process timeseries query filter.
        /// </summary>
        /// <remarks></remarks>
        private void ShowAndProcessTimeseriesQueryFilter()
        {
            if (timeseriesDataset != null)
            {
                if (timeseriesDataset.TotalRecords.Count > 0)
                {
                    timeseriesDataset.QueryFilter.DataSource = timeseriesDataset.TotalRecords;
                    if (timeseriesDataset.QueryFilter.ShowFilterDialog())
                    {
                        List<TimeseriesRecord> list = timeseriesDataset.QueryFilter.Execute();
                        if (list == null)
                        { list = new List<TimeseriesRecord>(); }
                        timeseriesDataset.FilteredRecords = list;
                        textboxDataFilterSummary.Text = timeseriesDataset.QueryFilter.Summary;
                        _TimeseriesFeatures = null;
                        _TimeseriesLayer = null;
                        BuildMapLayers(false);
                    }
                    timeseriesDataset.QueryFilter.DataSource = null;
                }
            }


        }
        /// <summary>
        /// Creates the and initialize map control.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        private MapControl CreateAndInitializeMapControl()
        {
            MapControl c = new MapControl();
            c.BackColor = System.Drawing.Color.White;
            c.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            c.MapBackgroundColor = System.Drawing.Color.White;
            c.TabIndex = 0;

            // Connect the MouseDoubleClick event handler
            c.MouseDoubleClick += new System.Windows.Forms.MouseEventHandler(this.mapControl_MouseDoubleClick);
            // Connect the MouseClick event handler
            c.MouseClick += new System.Windows.Forms.MouseEventHandler(this.mapControl_MouseClick);
            // Connect the MouseMove event handler
            c.MouseMove += new System.Windows.Forms.MouseEventHandler(this.mapControl_MouseMove);
            
            return c;
        }
        /// <summary>
        /// Creates the and initialize index map control.
        /// </summary>
        /// <param name="sourceMap">The source map.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private IndexMapControl CreateAndInitializeIndexMapControl(MapControl sourceMap)
        {
            IndexMapControl c = new IndexMapControl();
            c.SourceMap = sourceMap;
            c.SuppressMapImageUpdate = false;
            c.TabIndex = 0;
            return c;
        }
        /// <summary>
        /// Edits the symbology.
        /// </summary>
        /// <remarks></remarks>
        private void EditSymbology()
        {
            if (_SimData == null)
            { return; }

            switch (_SimData.SimulationType)
            {
                case 1:
                    if (_EndpointSymbologyDialog == null)
                    { return; }

                    if (_EndpointSymbologyDialog.ShowMapSymbologyDialog())
                    {
                        if (_EndpointSymbologyDialog.EndpointOption != _CurrentEndpointFeatures)
                        {
                            _EndpointLayer = null;
                            _EndpointFeatures = null;
                        }

                        if (_EndpointLayer != null)
                        {
                            _EndpointSymbologyDialog.SetRenderer(_EndpointLayer);
                            // Select the QuickView rendering option, which also causes the 
                            // map to refresh.
                            SelectQuickViewEndpoints(toolStripComboBoxQuickView.SelectedIndex);
                            BuildMapLegend();
                        }
                        else
                        {
                            BuildMapLayers(false);
                        }


                    }
                    break;
                case 2:
                    if (_PathlineSymbologyDialog == null)
                    { return; }
                    if (_PathlineSymbologyDialog.ShowSymbologyDialog())
                    {
                        if (_PathlineLayer != null)
                        {
                            _PathlineSymbologyDialog.SetRenderer(_PathlineLayer);
                        }

                        // Select the QuickView rendering option, which also causes the 
                        // map to refresh.
                        SelectQuickViewPathlines(toolStripComboBoxQuickView.SelectedIndex);

                        BuildMapLegend();

                    }
                    break;
                case 3:
                    if (_TimeseriesSymbologyDialog == null)
                    { return; }
                    if (_TimeseriesSymbologyDialog.ShowSymbologyDialog())
                    {
                        if (_TimeseriesLayer != null)
                        {
                            _TimeseriesSymbologyDialog.SetRenderer(_TimeseriesLayer);
                        }

                        // Select the QuickView rendering option, which also causes the 
                        // map to refresh.
                        SelectQuickViewTimeseries(toolStripComboBoxQuickView.SelectedIndex);

                        BuildMapLegend();

                    }

                    break;
                default:
                    throw new Exception("Invalid simulation type.");
            }

        }
        /// <summary>
        /// Edits the contour layer.
        /// </summary>
        /// <remarks></remarks>
        private void EditContourLayer()
        {
            if (_ContourEngineData != null)
            {
                ModflowOutputContoursEditDialog dialog = new ModflowOutputContoursEditDialog();
                dialog.ContourData = _ContourEngineData;
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    // Generate contours
                    LayerDataRecord<float> headLayerRecord = _ContourEngineData.ContourSourceFile.GetRecordAsSingle(_ContourEngineData.SelectedDataLayer);
                    if (headLayerRecord != null)
                    {
                        GenerateAndBuildContourLayer(headLayerRecord.DataArray, _ModelGrid);
                    }
                    else
                    {
                        _CurrentContourMapLayer = null;
                    }
                    BuildMapLayers(false);
                    BuildMapLegend();
                }
            }
        }
        /// <summary>
        /// Enables the time point navigation controls.
        /// </summary>
        /// <param name="visible">if set to <c>true</c> [visible].</param>
        /// <param name="enabled">if set to <c>true</c> [enabled].</param>
        /// <remarks></remarks>
        private void EnableTimePointNavigationControls(bool visible, bool enabled)
        {
            //toolStripLabelQuickView.Visible = visible;
            //toolStripComboBoxQuickView.Visible = visible;
            toolStripButtonFirstTimePoint.Visible = visible;
            toolStripButtonPreviousTimePoint.Visible = visible;
            toolStripButtonNextTimePoint.Visible = visible;
            toolStripButtonMoveLastTimePoint.Visible = visible;
            toolStripButtonAnimateTimepoints.Visible = visible;
            toolStripButtonContinuousLoop.Visible = visible;
            toolStripComboBoxAnimationInterval.Visible = visible;
            //toolStripComboBoxQuickView.Enabled = enabled;
            toolStripButtonNextTimePoint.Enabled = enabled;
            toolStripButtonPreviousTimePoint.Enabled = enabled;
        }
        /// <summary>
        /// Initializes the quick view timeseries.
        /// </summary>
        /// <param name="timePointList">The time point list.</param>
        /// <remarks></remarks>
        private void InitializeQuickViewTimeseries(Dictionary<int,double> timePointList)
        {
            string s;
            toolStripComboBoxQuickView.Items.Clear();
            toolStripComboBoxQuickView.Items.Add("Show all available time points");
            int[] keys = timePointList.Keys.ToArray<int>();
            for (int i = 0; i < keys.Length; i++)
            {
                s = "Time point " + keys[i].ToString() + " : Time = " + timePointList[keys[i]].ToString();
                toolStripComboBoxQuickView.Items.Add(s);
            }
            if (toolStripComboBoxQuickView.Items.Count  > 0 )
            {
                toolStripComboBoxQuickView.SelectedIndex = 0;
            }
        }
        /// <summary>
        /// Initializes the quick view endpoints.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <param name="fieldName">Name of the field.</param>
        /// <remarks></remarks>
        private void InitializeQuickViewEndpoints(List<int> values, string fieldName)
        {
            string s = fieldName.ToLower();
            toolStripComboBoxQuickView.Items.Clear();
            toolStripComboBoxQuickView.Items.Add("All available " + fieldName + " values");
            
            if ((s == "finalzone") || (s == "initzone"))
            {
                toolStripComboBoxQuickView.Items.Add("All available " + fieldName + " values except zone 1");
            }

            for (int i = 0; i < values.Count; i++)
            {
                toolStripComboBoxQuickView.Items.Add(fieldName + " = " + values[i].ToString());
            }

            toolStripComboBoxQuickView.SelectedIndex = 0;

        }
        /// <summary>
        /// Initializes the quick view pathlines.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <param name="fieldName">Name of the field.</param>
        /// <remarks></remarks>
        private void InitializeQuickViewPathlines(List<int> values, string fieldName)
        {
            string s = fieldName.ToLower();
            toolStripComboBoxQuickView.Items.Clear();
            toolStripComboBoxQuickView.Items.Add("All available " + fieldName + " values");

            if ((s == "finalzone") || (s == "initzone"))
            {
                toolStripComboBoxQuickView.Items.Add("All available " + fieldName + " values except zone 1");
            }

            for (int i = 0; i < values.Count; i++)
            {
                toolStripComboBoxQuickView.Items.Add(fieldName + " = " + values[i].ToString());
            }

            toolStripComboBoxQuickView.SelectedIndex = 0;

        }
        /// <summary>
        /// Selects the quick view timeseries.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <remarks></remarks>
        private void SelectQuickViewTimeseries(int index)
        {
            if (_TimeseriesLayer == null)
            { return; }

            _TimeseriesLayer.Renderer.MaskValues.Clear();
            _TimeseriesLayer.Renderer.MaskField = "Time";
            _TimeseriesLayer.Renderer.IncludeMaskValues = true;
            _TimeseriesLayer.Renderer.UseMask = false;
            if (index > 0)
            {
                _TimeseriesLayer.Renderer.UseMask = true;
                _TimeseriesLayer.Renderer.MaskValues.Add(_QuickViewTimeseriesValues[index - 1]);
            }

            // Redraw the map to reflect the rendering change
            mapControl.Refresh();

        }
        /// <summary>
        /// Selects the quick view endpoints.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <remarks></remarks>
        private void SelectQuickViewEndpoints(int index)
        {
            if (_EndpointLayer == null)
            { return; }

            _EndpointLayer.Renderer.MaskValues.Clear();
            if (_SimData.TrackingDirection == 1)
            {
                _EndpointLayer.Renderer.MaskField = "FinalZone";
            }
            else
            {
                _EndpointLayer.Renderer.MaskField = "InitZone";

            }
            _EndpointLayer.Renderer.IncludeMaskValues = false;
            _EndpointLayer.Renderer.UseMask = false;

            if (index > 0)
            {
                if (_EndpointLayer.Renderer.MaskField == "FinalZone" || _EndpointLayer.Renderer.MaskField == "InitZone")
                {
                    if (index == 1)
                    {
                        _EndpointLayer.Renderer.IncludeMaskValues = false;
                        _EndpointLayer.Renderer.UseMask = true;
                        _EndpointLayer.Renderer.MaskValues.Add(1);
                    }
                    else
                    {
                        _EndpointLayer.Renderer.IncludeMaskValues = true;
                        _EndpointLayer.Renderer.UseMask = true;
                        _EndpointLayer.Renderer.MaskValues.Add(Convert.ToDouble(_QuickViewEndpointValues[index-2]));
                    }
                }
            }

            // Redraw the map to reflect the rendering change
            mapControl.Refresh();

        }
        /// <summary>
        /// Selects the quick view pathlines.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <remarks></remarks>
        private void SelectQuickViewPathlines(int index)
        {
            if (_PathlineLayer == null)
            { return; }
            
            _PathlineLayer.Renderer.MaskValues.Clear();
            if (_SimData.TrackingDirection == 1)
            {
                _PathlineLayer.Renderer.MaskField = "FinalZone";
            }
            else
            {
                _PathlineLayer.Renderer.MaskField = "InitZone";

            }
            _PathlineLayer.Renderer.IncludeMaskValues = false;
            _PathlineLayer.Renderer.UseMask = false;

            if (index > 0)
            {
                if (_PathlineLayer.Renderer.MaskField == "FinalZone" || _PathlineLayer.Renderer.MaskField == "InitZone")
                {
                    if (index == 1)
                    {
                        _PathlineLayer.Renderer.IncludeMaskValues = false;
                        _PathlineLayer.Renderer.UseMask = true;
                        _PathlineLayer.Renderer.MaskValues.Add(1);
                    }
                    else
                    {
                        _PathlineLayer.Renderer.IncludeMaskValues = true;
                        _PathlineLayer.Renderer.UseMask = true;
                        _PathlineLayer.Renderer.MaskValues.Add(Convert.ToDouble(_QuickViewPathlineValues[index - 2]));
                    }
                }
            }

            // Redraw the map to reflect the rendering change
            mapControl.Refresh();

        }
        /// <summary>
        /// Starts the time point animation.
        /// </summary>
        /// <remarks></remarks>
        private void StartTimePointAnimation()
        {
            if (_AnimateTimePointTimer == null)
            {
                _AnimateTimePointTimer = new Timer();
                _AnimateTimePointTimer.Tick += new EventHandler(AnimateTimePointHandler_Tick);
            }
            SetAnimationInterval(_AnimateTimePointTimer, toolStripComboBoxAnimationInterval.Text);
            toolStripButtonAnimateTimepoints.Image = global::ModpathOutputExaminer.Properties.Resources.StopAnimation;
            _AnimateTimePointTimer.Start();
        }
        /// <summary>
        /// Sets the animation interval.
        /// </summary>
        /// <param name="timer">The timer.</param>
        /// <param name="text">The text.</param>
        /// <remarks></remarks>
        private void SetAnimationInterval(Timer timer, string text)
        {
            if (timer != null)
            {
                double frameinterval = double.Parse(text);
                frameinterval = Math.Round(frameinterval * 1000.0, 0);
                int interval = Convert.ToInt32(frameinterval);
                timer.Interval = interval;
            }
        }
        /// <summary>
        /// Stops the time point animation.
        /// </summary>
        /// <remarks></remarks>
        private void StopTimePointAnimation()
        {
            if (_AnimateTimePointTimer != null)
            {
                toolStripButtonAnimateTimepoints.Image = global::ModpathOutputExaminer.Properties.Resources.MoveForwardAnimate;
                _AnimateTimePointTimer.Stop();
            }
            
        }
        /// <summary>
        /// Gets a value indicating whether [animation is running].
        /// </summary>
        /// <remarks></remarks>
        private bool AnimationIsRunning
        {
            get
            {
                if (_AnimateTimePointTimer == null)
                { return false; }
                return _AnimateTimePointTimer.Enabled;
            }
        }
        /// <summary>
        /// Loads the basemap.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <remarks></remarks>
        private void LoadBasemap(string filename)
        {
            string basemapDirectory = "";
            string localBasemapName = "";
            Basemap bm = null;
            _Basemap = null;
            try
            {
                basemapDirectory = Path.GetDirectoryName(filename);
                localBasemapName = Path.GetFileName(filename);
                bm = Basemap.Read(basemapDirectory, localBasemapName);
                _Basemap = bm;
                _BasemapLayers = _Basemap.CreateBasemapLayers();
            }
            catch (Exception)
            {
                _Basemap = null;
                _BasemapLayers = null;
                System.Windows.Forms.MessageBox.Show("The basemap could not be loaded.");
            }

            BuildMapLayers(false);
            BuildMapLegend();

        }
        /// <summary>
        /// Sets the basemap visibility.
        /// </summary>
        /// <param name="visible">if set to <c>true</c> [visible].</param>
        /// <remarks></remarks>
        private void SetBasemapVisibility(bool visible)
        {
            _ShowBasemap = visible;
            this.Refresh();
            if (_Basemap != null)
            {
                BuildMapLayers(false);
                mapControl.Refresh();
                indexMapControl.UpdateMapImage();
            }
        }
        /// <summary>
        /// Exports the shapefile.
        /// </summary>
        /// <param name="dataset">The dataset.</param>
        /// <param name="features">The features.</param>
        /// <remarks></remarks>
        private void ExportShapefile(SimulationData dataset, FeatureCollection features)
        {
            if (dataset == null)
            { return; }

            if (features == null)
            { throw new ArgumentNullException("features"); }

            if (features.Count == 0)
            {
                throw new ArgumentException("The feature collection is empty.");
            }

            string filename = "";
            if (dataset.SimulationType == 1)
            {
                filename = dataset.SimulationName + "_endpoints";
            }
            else if (dataset.SimulationType == 2)
            {
                filename = dataset.SimulationName + "_pathlines";
            }
            else if (dataset.SimulationType == 3)
            {
                filename = dataset.SimulationName + "_timeseries";
            }
            else
            { throw new ArgumentException("The specified simulation type is invalid."); }

            if (!string.IsNullOrEmpty(filename))
            {
                try
                {
                    toolStripStatusMessage.Text = "Exporting shapefile: " + filename + "  ...";
                    statusStripMain.Refresh();
                    EsriShapefileIO.Export(features, dataset.WorkingDirectory, filename);
                    toolStripStatusMessage.Text = "Shapefile was exported: " + filename;
                }
                catch (Exception e)
                {
                    toolStripStatusMessage.Text = "Shapefile export failed.";
                    throw new Exception("Failed to export shapefile: " + filename, e);
                }
            }


        }
        /// <summary>
        /// Updates the status bar location info.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <remarks></remarks>
        private void UpdateStatusBarLocationInfo(int x, int y)
        {
            ICoordinate coord = mapControl.ToMapPoint(x, y);
            string mouseLocation = "";
            string cellCoord = "";

            if (mapControl.LayerCount > 0)
            { 
                mouseLocation = "X: " + coord.X.ToString("#.00") + "  Y: " + coord.Y.ToString("#.00"); 
            }

            if (_ModelGrid != null)
            {
                GridCell cell = _ModelGrid.FindRowColumn(coord);
                // Process the cell and contour information if the location is within the grid.
                if (cell != null)
                {
                    // Update status bar with current grid cell and contour data
                    cellCoord = "Row " + cell.Row.ToString() + "  Col " + cell.Column.ToString();
                }
            }

            toolStripStatusMapXyLocation.Text = mouseLocation;
            toolStripStatusMapCellLocation.Text = cellCoord;

        }
        /// <summary>
        /// Clears the map legend.
        /// </summary>
        /// <remarks></remarks>
        private void ClearMapLegend()
        {
            legendParticles.Clear();
            legendParticles.LegendTitle = "";
        }
        /// <summary>
        /// Builds the map legend.
        /// </summary>
        /// <remarks></remarks>
        private void BuildMapLegend()
        {
            ClearMapLegend();
            Collection<GraphicLayer> legendItems = new Collection<GraphicLayer>();

            // Add model grid outline
            if (_GridOutlineMapLayer != null)
            {
                legendItems.Add(_GridOutlineMapLayer);
            }

            // Add model grid interior lines
            if (_GridlinesMapLayer != null)
            {
                legendItems.Add(_GridlinesMapLayer);
            }

            // Add head contour layer
            if (_CurrentContourMapLayer != null)
            {
                legendItems.Add(_CurrentContourMapLayer);
            }

            // Add particle layer
            if (_SimData != null)
            {
                if (_SimData.SimulationType == 1)
                {
                    if (_EndpointLayer != null)
                    {
                        if (_SimData.TrackingDirection == 1)
                        {
                            legendParticles.LegendTitle = "Forward Tracked Endpoints";
                        }
                        else if (_SimData.TrackingDirection == 2)
                        {
                            legendParticles.LegendTitle = "Backward Tracked Endpoints";
                        }
                        legendItems.Add(_EndpointLayer);
                    }
                }
                else if (_SimData.SimulationType == 2)
                {
                    if (_SimData.TrackingDirection == 1)
                    {
                        legendParticles.LegendTitle = "Forward Tracked Pathlines";
                    }
                    else if (_SimData.TrackingDirection == 2)
                    {
                        legendParticles.LegendTitle = "Backward Tracked Pathlines";
                    }
                    legendItems.Add(_PathlineLayer);
                }
                else if (_SimData.SimulationType == 3)
                {
                    if (_SimData.TrackingDirection == 1)
                    {
                        legendParticles.LegendTitle = "Forward Tracked Timeseries";
                    }
                    else if (_SimData.TrackingDirection == 2)
                    {
                        legendParticles.LegendTitle = "Backward Tracked Timeseries";
                    }
                    legendItems.Add(_TimeseriesLayer);
                }

            }

            // Add basemap layers
            if (_BasemapLayers != null)
            {
                for (int i = 0; i < _BasemapLayers.Count; i++)
                {
                    legendItems.Add(_BasemapLayers[i]);
                }
            }

            if ((_SimData == null) && (_BasemapLayers!=null))
            {
                legendParticles.LegendTitle = "Basemap";
            }
            legendParticles.AddItems(legendItems);
        }
        /// <summary>
        /// Prints the map.
        /// </summary>
        /// <param name="showPrintPreview">if set to <c>true</c> [show print preview].</param>
        /// <remarks></remarks>
        private void PrintMap(bool showPrintPreview)
        {
            // select the printer
            System.Windows.Forms.PrintDialog printDialog = new PrintDialog();
            // set the print dialog to the current application printer and printer settings
            // if one exists.
            if (_PrinterSettings != null)
            { printDialog.PrinterSettings = _PrinterSettings; }

            // display print dialog. If the dialog returns OK then save the printer settings,
            // construct the print document, and display the print preview dialog.
            if (printDialog.ShowDialog() == DialogResult.OK)
            {
                _PrinterSettings = printDialog.PrinterSettings;
                MapIO mapIO = new MapIO();
                mapIO.Map = mapControl;
                mapIO.Print(_PrinterSettings, this, showPrintPreview);
            }
        }
        /// <summary>
        /// Prints the PDF.
        /// </summary>
        /// <param name="pdfFilename">The PDF filename.</param>
        /// <remarks></remarks>
        private void PrintPDF(string pdfFilename)
        {
            PrintPdfDialog printPdfDialog = new PrintPdfDialog();
            printPdfDialog.Filename = pdfFilename;
            if (printPdfDialog.ShowDialog() == DialogResult.OK)
            {
                // Initialize page settings
                System.Windows.Forms.PageSetupDialog pageSetupDialog = new PageSetupDialog();
                if (_PdfPageSettings == null)
                { _PdfPageSettings = new System.Drawing.Printing.PageSettings(); }

                // Set pageSetupDialog page settings
                pageSetupDialog.PageSettings = _PdfPageSettings;

                if (pageSetupDialog.ShowDialog() == DialogResult.OK)
                {
                    try
                    {
                        string heading = "";
                        if (_SimData != null)
                        {
                            heading = _SimData.SimulationFilePath;
                        }
                        _PdfPageSettings = pageSetupDialog.PageSettings;
                        MapIO mapIO = new MapIO();
                        mapIO.Map = mapControl;
                        mapIO.ExportPDF(printPdfDialog.Filename, _PdfPageSettings, heading, printPdfDialog.Title, printPdfDialog.Description, false, 10);
                    }
                    catch (Exception ex)
                    {
                        if (ex is ArgumentException)
                        {
                            ArgumentException argEx = ex as ArgumentException;
                            MessageBox.Show(argEx.Message);
                        }
                        else
                        { MessageBox.Show(ex.Message); }

                        return;
                    }

                }
            }
        }
        /// <summary>
        /// Transforms the features from coordinates relative to the model grid to global coordinates.
        /// </summary>
        /// <param name="features">The feature collection containing the features to transform.</param>
        /// <param name="modelGrid">The model grid.</param>
        /// <param name="fromRelativeToGlobal">if set to <c>true</c> [from relative to global].</param>
        /// <remarks></remarks>
        private void TransformFeatures(FeatureCollection features, CellCenteredArealGrid modelGrid, bool fromRelativeToGlobal)
        {
            if (modelGrid == null || features == null)
            { return; }
            
            if (modelGrid.Angle != 0.0 || modelGrid.OriginX != 0.0 || modelGrid.OriginY != 0.0)
            {
                if (fromRelativeToGlobal)
                {
                    foreach (Feature item in features)
                    {
                        modelGrid.TransformRelativeToGlobal(item.Geometry);
                    }
                }
                else
                {
                    foreach (Feature item in features)
                    {
                        modelGrid.TransformGlobalToRelative(item.Geometry);
                    }
                }
            }

        }
        /// <summary>
        /// Edits the metadata.
        /// </summary>
        /// <remarks></remarks>
        private void EditMetadata()
        {
            if (_Metadata != null)
            {
                ModflowMetadaEditDialog dialog = new ModflowMetadaEditDialog();
                dialog.Metadata = _Metadata;
                if (_Basemap != null)
                {
                    dialog.CurrentBasemapFile = Path.Combine(_Basemap.BasemapDirectory, _Basemap.Filename);
                }
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    // Save the metadata
                    ModflowMetadata.Write(_Metadata.Filename, _Metadata);

                    // Remove the basemap if one is loaded
                    if (_Basemap != null)
                    {
                        _Basemap = null;
                        _BasemapLayers.Clear();
                    }

                    // Re-load the current dataset
                    OpenDataset(_SimData.SimulationFilePath);

                }
            }
        }
        /// <summary>
        /// Generates the and build contour layer.
        /// </summary>
        /// <param name="buffer">The buffer.</param>
        /// <param name="modelGrid">The model grid.</param>
        /// <remarks></remarks>
        private void GenerateAndBuildContourLayer(Array2d<float> buffer, CellCenteredArealGrid modelGrid)
        {
            ContourLineList contourList = GenerateContours(buffer, modelGrid);
            FeatureLayer contourLayer = BuildContourLayer(contourList);
            bool layerVisible = false;
            if (_CurrentContourMapLayer != null)
            {
                layerVisible = _CurrentContourMapLayer.Visible;
            }
            contourLayer.LayerName = "Head, Layer " + _ContourEngineData.SelectedDataLayer.Layer.ToString() + " (Period " + _ContourEngineData.SelectedDataLayer.StressPeriod.ToString() + ", Step " + _ContourEngineData.SelectedDataLayer.TimeStep.ToString() + ")";
            _CurrentContourMapLayer = contourLayer;
            _CurrentContourMapLayer.Visible = layerVisible;
        }
        /// <summary>
        /// Generates the contours.
        /// </summary>
        /// <param name="buffer">The buffer.</param>
        /// <param name="modelGrid">The model grid.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private ContourLineList GenerateContours(Array2d<float> buffer, CellCenteredArealGrid modelGrid)
        {
            if (buffer == null)
                throw new ArgumentNullException();
            if ((buffer.RowCount != modelGrid.RowCount) || (buffer.ColumnCount != modelGrid.ColumnCount))
                throw new ArgumentException("Array does not match model grid dimensions.");

            ContourEngine ce = new ContourEngine(modelGrid);

            ce.UseDefaultNoDataRange = false;
            foreach (float excludedValue in _ContourEngineData.ExcludedValues)
            {
                ce.ExcludedValues.Add(excludedValue);
            }
            ce.LayerArray = buffer;
            float refContour = _ContourEngineData.ReferenceContour;

            float conInterval;

            switch (_ContourEngineData.ContourIntervalOption)
            {
                case ContourIntervalOption.AutomaticConstantInterval:
                    conInterval = ce.ComputeContourInterval();
                    ce.ContourLevels = ce.GenerateConstantIntervals(conInterval, refContour);
                    break;
                case ContourIntervalOption.SpecifiedConstantInterval:
                    conInterval = _ContourEngineData.ConstantContourInterval;
                    ce.ContourLevels = ce.GenerateConstantIntervals(conInterval, refContour);
                    break;
                case ContourIntervalOption.SpecifiedContourLevels:
                    ce.ContourLevels = _ContourEngineData.ContourLevels;
                    break;
                default:
                    break;
            }

            ContourLineList conLineList = ce.CreateContours();
            return conLineList;

        }
        /// <summary>
        /// Builds the contour layer.
        /// </summary>
        /// <param name="contourList">The contour list.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private FeatureLayer BuildContourLayer(ContourLineList contourList)
        {
            if (contourList == null)
                throw new ArgumentNullException("The specified contour list does not exist.");

            FeatureLayer contourLayer = new FeatureLayer(LayerGeometryType.Line);
            ILineSymbol symbol = ((ISingleSymbolRenderer)(contourLayer.Renderer)).Symbol as ILineSymbol;
            symbol.Color = _ContourEngineData.ContourColor;
            symbol.Width = Convert.ToSingle(_ContourEngineData.ContourLineWidth);
            for (int i = 0; i < contourList.Count; i++)
            {
                IAttributesTable attributes = new AttributesTable();
                attributes.AddAttribute("Value", contourList[i].ContourLevel);
                contourLayer.AddFeature(contourList[i].Contour as IGeometry, attributes);
            }

            contourLayer.LayerName = "Current Data Contours";
            return contourLayer;
        }
        /// <summary>
        /// Saves the basemap as.
        /// </summary>
        /// <remarks></remarks>
        private void SaveBasemapAs()
        {
            if (_Basemap == null)
            {
                return;
            }
            if (_SimData == null)
            {
                return;
            }

            SaveFileDialog dialog = new SaveFileDialog();
            dialog.InitialDirectory = _SimData.WorkingDirectory;
            dialog.FileName = "";
            dialog.DefaultExt = "basemap";
            dialog.Filter = "*.basemap (Basemap files)|*.basemap|*.* (All files)|*.*";
            dialog.AddExtension = true;
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                _Basemap.BasemapDirectory = Path.GetDirectoryName(dialog.FileName);
                _Basemap.Filename = Path.GetFileName(dialog.FileName);
                Basemap.Write(_Basemap);
            }
        }
        /// <summary>
        /// Finds the contour line hit.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <remarks></remarks>
        private void FindContourLineHit(int x, int y)
        {
            _HotFeature = null;
            ICoordinate c = mapControl.ToMapPoint(x, y);
            IPoint pt = new USGS.Puma.NTS.Geometries.Point(c);
            double tol = (mapControl.ViewportExtent.Width / Convert.ToDouble(mapControl.ViewportSize.Width)) * 2.0;
            if (_CurrentContourMapLayer != null)
            {
                if (_CurrentContourMapLayer.Visible)
                {
                    for (int i = 0; i < _CurrentContourMapLayer.FeatureCount; i++)
                    {
                        Feature f = _CurrentContourMapLayer.GetFeature(i);
                        if (f.Geometry.IsWithinDistance(pt as IGeometry, tol))
                        {
                            _HotFeature = f;
                            break;
                        }
                    }
                }
            }
        }
        /// <summary>
        /// Shows the map tip.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <remarks></remarks>
        private void ShowMapTip(int x, int y)
        {
            if (_HotFeature != null)
            {
                string slevel = null;
                float value = Convert.ToSingle(_HotFeature.Attributes["Value"]);
                if (value >= 1.0f && value < 10000.0f)
                {
                    slevel = "Value = " + value.ToString("#.##");
                }
                else if (value <= -1.0f && value > -10000.0f)
                {
                    slevel = "Value = " + value.ToString("#.##");
                }
                else
                {
                    slevel = "Value = " + value.ToString("#.###E+00");
                }

                _MapTip.Show(slevel, mapControl, x + 12, y - 15, 2000);
            }
        }

        #endregion




















    }
}
