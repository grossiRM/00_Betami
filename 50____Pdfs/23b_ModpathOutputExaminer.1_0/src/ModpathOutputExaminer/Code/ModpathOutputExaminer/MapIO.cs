using System;
using System.Collections.Generic;
using System.Text;
using GeoAPI.Geometries;
using System.Windows.Forms;
using System.Drawing.Printing;
using dbAutoTrack.PDFWriter;
using dbAutoTrack.PDFWriter.Graphics;
using dbAutoTrack.PDFWriter.Form;
using System.IO;
using System.Drawing;

namespace USGS.Puma.UI.MapViewer
{
    public class MapIO
    {
        private IMap _Map;
        public IMap Map
        {
            get { return _Map; }
            set { _Map = value; }
        }

        public void ExportPDF(string filename, PageSettings pageSettings)
        {
            ExportPDF(filename, pageSettings, "", "", "", false, 10);
        }
        public void ExportPDF(string filename, PageSettings pageSettings, string dataSourceName, string title, string description, bool drawFrame, int fontSize)
        {
            if (_Map == null)
                throw new ArgumentNullException("Map");
            if (pageSettings == null)
                throw new ArgumentNullException("pageSettings");

            System.Drawing.Image img = null;

            // get page size and check validity
            PageSize pageSize = new PageSize();
            try
            {
                pageSize = GetPdfPageSize(pageSettings.PaperSize.Kind);
            }
            catch (Exception)
            {
                throw new ArgumentException("The PDF exporter does not support the kind of paper that was selected.");
            }

            // Process the PDF file
            try
            {
                // The units of the margins are hundredths of an inch. Convert the margin values from hundredths of an inch 
                // to points (0.72 points = 0.01 inch)
                float leftMargin = pageSettings.Margins.Left * 0.72f;
                float rightMargin = pageSettings.Margins.Right * 0.72f;
                float topMargin = pageSettings.Margins.Top * 0.72f;
                float bottomMargin = pageSettings.Margins.Bottom * 0.72f;

                //Initialize PDF Document
                Document _document = new Document();

                _document.Title = title;

                //Compression is set to true by default.
                //_document.Compress = false;

                //Initialize new page with PageSize Letter and the portrait/landscape
                //orientation specified in the PageSettingsDialog.
                Page page1 = null;

                if (pageSettings.Landscape)
                { page1 = new Page(pageSize, PageOrientation.Landscape); }
                else
                { page1 = new Page(pageSize); }

                float pageWidth = page1.Width;
                float pageHeight = page1.Height;

                //Get the PDFGraphics object for drawing to the page.
                PDFGraphics graphics1 = page1.Graphics;

                // Write the header lines
                PDFFont pdfFont = new PDFFont(StandardFonts.TimesRoman, FontStyle.Regular);
                float textBottom = topMargin;
                bool addPadding = false;
                if (!String.IsNullOrEmpty(dataSourceName.Trim()))
                {
                    addPadding = true;
                    textBottom += fontSize;
                    graphics1.DrawString(leftMargin, textBottom, dataSourceName, pdfFont, fontSize);
                }
                if (!String.IsNullOrEmpty(title.Trim()))
                {
                    addPadding = true;
                    textBottom += fontSize + 2;
                    graphics1.DrawString(leftMargin, textBottom, title, pdfFont, fontSize);
                }
                if (!String.IsNullOrEmpty(description.Trim()))
                {
                    addPadding = true;
                    textBottom += fontSize + 2;
                    graphics1.DrawString(leftMargin, textBottom, description, pdfFont, fontSize);
                }
                if (addPadding)
                    textBottom += 6;

                //Find the aspect ratio of the MapExtent.
                double width = 500;
                double height = 500;
                if (_Map.MapExtent.Width > _Map.MapExtent.Height)
                { height = width * _Map.MapExtent.Height / _Map.MapExtent.Width; }
                else
                { width = height * _Map.MapExtent.Width / _Map.MapExtent.Height; }

                // Render the map image with the same aspect ratio as the MapExtent.
                System.Drawing.Size size = new Size(Convert.ToInt32(width), Convert.ToInt32(height));
                img = _Map.RenderAsImage(size);

                // Calculate the dimensions of the frame rectangle for the map image.
                pageWidth = pageWidth - leftMargin - rightMargin;
                pageHeight = pageHeight - textBottom - bottomMargin;
                float pageAspect = pageWidth / pageHeight;
                float mapExtentAspect = Convert.ToSingle(width / height);
                float frameHeight;
                float frameWidth;
                if (pageAspect > mapExtentAspect)
                {
                    frameHeight = pageHeight;
                    frameWidth = pageWidth;
                }
                else
                {
                    frameWidth = pageWidth;
                    frameHeight = frameWidth / mapExtentAspect;
                }

                // Draw the image. Use SizeMode.Zoom so that the image is scaled to fill the target rectangle defined by pageWidth x pageHeight.
                // The Zoom mode takes care of all the scaling issues associated with going from pixels to points.
                // graphics1.DrawImage(img, 36, 36, page1.Width - 72, page1.Height - 72, PictureAlignment.TopLeft, SizeMode.Zoom, new Border(1f, RGBColor.Red, LineStyle.Solid));
                if (drawFrame)
                { graphics1.DrawImage(img, leftMargin, textBottom, pageWidth, pageHeight, PictureAlignment.Center, SizeMode.Zoom, new Border(1f, RGBColor.Red, LineStyle.Solid)); }
                else
                { graphics1.DrawImage(img, leftMargin, textBottom, pageWidth, pageHeight, PictureAlignment.Center, SizeMode.Zoom); }

                // Add the page to the Document
                _document.Pages.Add(page1);

                // Create a file stream and generate the PDF file.
                FileStream _fs = new FileStream(filename, System.IO.FileMode.Create, System.IO.FileAccess.Write);
                try
                {
                    //Generate PDF to the stream
                    _document.Generate(_fs);
                    MessageBox.Show(filename + " was generated successfully.");
                }
                finally
                {
                    _fs.Flush();
                    _fs.Close();
                }
            }
            finally
            {
                if (img != null)
                    img.Dispose();
            }


        }

        public void Print(System.Drawing.Printing.PrinterSettings printerSettings, System.Windows.Forms.Form owner, bool printPreview)
        {
            if (printerSettings == null)
                throw new ArgumentNullException("No printer settings were specified.");

            System.Drawing.Printing.PrintDocument pd = null;
            try
            {
                // create printer document and set the printer settings to selected printer
                pd = new System.Drawing.Printing.PrintDocument();
                pd.PrinterSettings = printerSettings;

                // subscribe to the printer document PrintPage event
                pd.PrintPage += new System.Drawing.Printing.PrintPageEventHandler(pd_PrintPage);

                if (printPreview)
                {
                    // create the print preview dialog
                    PrintPreviewDialog previewDialog = new PrintPreviewDialog();

                    // attach the printer document to the print preview control
                    previewDialog.Document = pd;

                    // Set size to 800x800. Set the startup location to the center of 
                    // the screen.
                    previewDialog.Width = 800;
                    previewDialog.Height = 800;
                    ((System.Windows.Forms.Form)previewDialog).StartPosition = FormStartPosition.CenterScreen;

                    // show the print preview dialog
                    if (owner == null)
                    { previewDialog.ShowDialog(); }
                    else
                    { previewDialog.ShowDialog(owner); }
                }
                else
                { pd.Print(); }

            }
            finally
            {
                pd.PrintPage -= pd_PrintPage;
            }

        }

        public void SaveImage(string filename, System.Drawing.Imaging.ImageFormat imageFormat, System.Drawing.Size size)
        {
            if (_Map == null)
                throw new ArgumentNullException("Map");
            if (String.IsNullOrEmpty(filename.Trim()))
                throw new ArgumentNullException("No filename was specified");

            System.Drawing.Image img = null;
            try
            {
                img = _Map.RenderAsImage(size);
                img.Save(filename, imageFormat);
            }
            finally
            {
                if (img != null)
                    img.Dispose();
            }
            
        }

        private PageSize GetPdfPageSize(System.Drawing.Printing.PaperKind kind)
        {
            switch (kind)
            {
                case System.Drawing.Printing.PaperKind.A2:
                    return PageSize.A2;
                case System.Drawing.Printing.PaperKind.A3:
                    return PageSize.A3;
                case System.Drawing.Printing.PaperKind.A4:
                    return PageSize.A4;
                case System.Drawing.Printing.PaperKind.A5:
                    return PageSize.A5;
                case System.Drawing.Printing.PaperKind.B4:
                    return PageSize.B4;
                case System.Drawing.Printing.PaperKind.Executive:
                    return PageSize.Executive;
                case System.Drawing.Printing.PaperKind.Folio:
                    return PageSize.Folio;
                case System.Drawing.Printing.PaperKind.Ledger:
                    return PageSize.Ledger;
                case System.Drawing.Printing.PaperKind.Legal:
                    return PageSize.Legal;
                case System.Drawing.Printing.PaperKind.Letter:
                    return PageSize.Letter;
                case System.Drawing.Printing.PaperKind.Quarto:
                    return PageSize.Quarto;
                case System.Drawing.Printing.PaperKind.Statement:
                    return PageSize.Statement;
                default:
                    throw new ArgumentException("The specified paper kind is not supported.");
            }
        }
        private void pd_PrintPage(object sender, System.Drawing.Printing.PrintPageEventArgs e)
        {
            if (_Map == null)
                throw new ArgumentNullException("Map");

            System.Drawing.Image img = null;
            try
            {
                int width = e.MarginBounds.Width;
                int height = e.MarginBounds.Height;
                img = _Map.RenderAsImage(new Size(width, height));
                e.Graphics.DrawImage(img, e.MarginBounds.X, e.MarginBounds.Y);
            }
            finally
            {
                if (img != null)
                    img.Dispose();
            }

        }

    }
}
