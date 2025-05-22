using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.NTS.Features;
using USGS.Puma.NTS.Geometries;
using GeoAPI.Geometries;
using System.Drawing;

namespace USGS.Puma.UI.MapViewer
{
    public class RendererHelper
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="symbol"></param>
        /// <returns></returns>
        public static ISymbolDrawingTool CreateDrawingTool(ISymbol symbol)
        {
            if (symbol == null)
                return null;

            if (symbol is ILineSymbol)
            {
                return CreateDrawingTool(symbol as ILineSymbol);
            }
            else if (symbol is ISolidFillSymbol)
            {
                return CreateDrawingTool(symbol as ISolidFillSymbol);
            }
            else if (symbol is IPointSymbol)
            {
                return CreateDrawingTool(symbol as IPointSymbol);
            }
            else
            { return null; }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="symbol"></param>
        /// <returns></returns>
        public static ISymbolDrawingTool CreateDrawingTool(ILineSymbol symbol)
        {
            if (symbol == null)
                return null;

            System.Drawing.Pen pen = new System.Drawing.Pen(symbol.Color, symbol.Width);
            pen.DashStyle = symbol.DashStyle;
            
            return new SymbolDrawingTool(null, pen, true, symbol as ISymbol);

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="symbol"></param>
        /// <returns></returns>
        public static ISymbolDrawingTool CreateDrawingTool(ISolidFillSymbol symbol)
        {
            if (symbol == null)
                return null;
            ISymbolDrawingTool tool = LineSymbol.CreateDrawingTool(symbol.Outline);
            tool.FillBrush = new System.Drawing.SolidBrush(symbol.Color);
            tool.EnableOutline = symbol.EnableOutline;
            tool.Symbol = symbol as ISymbol;
            return tool;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="symbol"></param>
        /// <returns></returns>
        public static ISymbolDrawingTool CreateDrawingTool(IPointSymbol symbol)
        {
            if (symbol == null)
            { return null; }
            ISymbolDrawingTool tool = new SymbolDrawingTool();

            tool.EnableOutline = true;
            System.Drawing.Pen pen = new Pen(symbol.Color, 1.0f);
            pen.DashStyle = System.Drawing.Drawing2D.DashStyle.Solid;
            tool.Pen = pen;
            if (symbol.IsFilled)
            { tool.FillBrush = new System.Drawing.SolidBrush(symbol.Color); }
            tool.Symbol = symbol as ISymbol;

            return tool;
        }
        
        /// <summary>
        /// 
        /// </summary>
        /// <param name="layers"></param>
        /// <param name="g"></param>
        /// <param name="vp"></param>
        public void Render(IGraphicLayerList layers, System.Drawing.Graphics g, Viewport vp)
        {
            if (layers != null)
            {
                IGraphicLayer layer = null;
                for (int i = layers.LayerCount - 1; i > -1; i--)
                {
                    layer = layers.GetLayer(i);
                    if (layer.Visible)
                    { 
                        RenderLayer(layer, g, vp); 
                    }
                }
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="symbol"></param>
        /// <param name="g"></param>
        /// <param name="size"></param>
        public void RenderLegendSymbol(ISymbol symbol, System.Drawing.Graphics g, System.Drawing.Size size)
        {
            Viewport vp = new Viewport(size);
            ISymbolDrawingTool tool = CreateDrawingTool(symbol);

            if (symbol is SimplePointSymbol)
            {
                PointF pt = new PointF(size.Width / 2.0f, size.Height / 2.0f);
                SimplePointSymbol ptSymbol = symbol as SimplePointSymbol;
                if (ptSymbol.SymbolType == PointSymbolTypes.Circle)
                {
                    vp.DrawCirclePoint(pt, g, tool.Pen, tool.FillBrush, ptSymbol.Size);
                }
                else if (ptSymbol.SymbolType == PointSymbolTypes.Square)
                {
                    vp.DrawSquarePoint(pt, g, tool.Pen, tool.FillBrush, ptSymbol.Size);
                }
            }
            else if (symbol is ISolidFillSymbol)
            {
                // Add code
            }
            else if (symbol is ILineSymbol)
            {
                PointF[] pts = new PointF[2];
                PointF center = new PointF(size.Width / 2.0f, size.Height / 2.0f);
                pts[0] = new PointF(center.X - 5.0f, center.Y);
                pts[1] = new PointF(center.X + 5.0f, center.Y);
                vp.DrawLineString(pts, g, tool.Pen);
            }

            (tool as IDisposable).Dispose();
         
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="layer"></param>
        /// <param name="g"></param>
        /// <param name="vp"></param>
        private void RenderVectorLayer(IFeatureLayer layer, Graphics g, Viewport vp)
        {
            if (layer == null)
                return;
            if (layer.Visible)
            {
                if (layer.Renderer != null)
                {
                    if (layer.GeometryType == LayerGeometryType.Line)
                    {
                        g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                        //g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.Default;
                    }
                    else
                    { g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.Default; }

                    if (layer.Renderer is ISingleSymbolRenderer)
                    { RenderSingleSymbolRenderer(layer, g, vp); }
                    else if (layer.Renderer is INumericValueRenderer)
                    { RenderNumericValueRenderer(layer, g, vp); }
                    else if (layer.Renderer is IColorRampRenderer)
                    { RenderColorRampRenderer(layer, g, vp); }
                    else if (layer.Renderer is IClassBreaksRenderer)
                    { throw new NotImplementedException("Render type IClassBreaksRenderer is not supported."); }
                    else
                    { throw new NotImplementedException("The specified render type is not supporte."); }
               
                }
            }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="layer"></param>
        /// <param name="g"></param>
        /// <param name="vp"></param>
        private void RenderLayer(IGraphicLayer layer, Graphics g, Viewport vp)
        {
            if (layer is IFeatureLayer)
            { RenderVectorLayer(layer as IFeatureLayer, g, vp); }
            else
            {
                // image layers not yet supported.
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="layer"></param>
        /// <param name="g"></param>
        /// <param name="vp"></param>
        private void RenderSingleSymbolRenderer(IFeatureLayer layer, Graphics g, Viewport vp)
        {
            if (layer != null)
            {
                ISingleSymbolRenderer renderer = layer.Renderer as ISingleSymbolRenderer;
                ISymbolDrawingTool tool = RendererHelper.CreateDrawingTool(renderer.Symbol);
                double maskValue = 0;

                try
                {
                    bool renderValue = true;
                    for (int i = 0; i < layer.FeatureCount; i++)
                    {
                        renderValue = true;
                        Feature f = layer.GetFeature(i);
                        if (renderer.UseMask)
                        {
                            if (renderer.UseMask)
                            {
                                maskValue = Convert.ToDouble(f.Attributes[renderer.MaskField]);
                                renderValue = (renderer.MaskValues.Contains(maskValue) == renderer.IncludeMaskValues);
                            }
                        }
                        if (renderValue)
                        {
                            DrawGeometry(f.Geometry, g, vp, tool, true);
                        }
                    }
                }
                catch (Exception)
                {
                    throw;
                }
                finally
                {
                    if (tool != null)
                        ((IDisposable)tool).Dispose();
                    if (renderer != null)
                        ((IDisposable)renderer).Dispose();
                }
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="layer"></param>
        /// <param name="g"></param>
        /// <param name="vp"></param>
        private void RenderNumericValueRenderer(IFeatureLayer layer, Graphics g, Viewport vp)
        {
            if (layer != null)
            {
                INumericValueRenderer renderer = layer.Renderer as INumericValueRenderer;
                List<ISymbolDrawingTool> tools = new List<ISymbolDrawingTool>();

                try
                {
                    for (int i = 0; i < renderer.ValueCount; i++)
                    {
                        tools.Add(RendererHelper.CreateDrawingTool(renderer.GetSymbol(i)));
                    }

                    Feature f = null;
                    double v;
                    double maskValue = 0;
                    int index;
                    bool renderValue = true;
                    for (int i = 0; i < layer.FeatureCount; i++)
                    {
                        renderValue = true;
                        f = layer.GetFeature(i);
                        v = Convert.ToDouble(f.Attributes[renderer.RenderField]);
                        if (renderer.UseMask)
                        {
                            maskValue = Convert.ToDouble(f.Attributes[renderer.MaskField]);
                            renderValue = (renderer.MaskValues.Contains(maskValue) == renderer.IncludeMaskValues);
                        }
                        if ((renderValue) && (renderer.ExcludedValues.Count > 0))
                        {
                            if (renderer.ExcludedValues.Contains(v))
                            {
                                renderValue = false;
                            }
                        }

                        if (renderValue)
                        {
                            index = renderer.IndexOf(v);
                            if (index >= 0)
                            {
                                DrawGeometry(f.Geometry, g, vp, tools[index], true);
                            }
                        }
                    }
                }
                catch (Exception)
                {
                    throw;
                }
                finally
                {
                    if (tools != null)
                    {
                        foreach (ISymbolDrawingTool tool in tools)
                        {
                            if (tool != null)
                                ((IDisposable)tool).Dispose();
                        }
                        tools.Clear();
                    }
                    if (renderer != null)
                        ((IDisposable)renderer).Dispose();
                }
            }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="layer"></param>
        /// <param name="g"></param>
        /// <param name="vp"></param>
        private void RenderColorRampRenderer(IFeatureLayer layer, Graphics g, Viewport vp)
        {
            if (layer != null)
            {
                ColorRampRenderer renderer = layer.Renderer as ColorRampRenderer;
                if (renderer.UseRenderArray && renderer.RenderArray == null)
                { return; }
                
                ISymbolDrawingTool tool = CreateDrawingTool(renderer.BaseSymbol);
                IFillSymbol fillSymbol = null;
                
                int alpha = renderer.BaseSymbol.Color.A;
                if (renderer.BaseSymbol is IFillSymbol)
                {
                    fillSymbol = (IFillSymbol)renderer.BaseSymbol;
                }
                System.Drawing.SolidBrush brush = null;
                if (tool.FillBrush != null)
                    brush = (System.Drawing.SolidBrush)tool.FillBrush;
                float p;
                double v;
                double maskV = 0.0;
                Feature f = null;
                object obj = null;

                try
                {
                    int count = layer.FeatureCount;
                    if (renderer.UseRenderArray)
                    {
                        if (renderer.RenderArray.Length < count)
                        { count = renderer.RenderArray.Length; }
                    }

                    bool renderValue = true;
                    for (int i = 0; i < count; i++)
                    {
                        renderValue = true;
                        f = layer.GetFeature(i);
                        if (renderer.UseRenderArray)
                        { v = renderer.RenderArray[i]; }
                        else
                        {
                            obj = f.Attributes[renderer.RenderField];
                            v = Convert.ToDouble(obj);
                        }

                        // Check MaskField if necessary
                        if (renderer.UseMask)
                        {
                            obj = f.Attributes[renderer.MaskField];
                            maskV = Convert.ToDouble(obj);
                            renderValue = (renderer.MaskValues.Contains(maskV) == renderer.IncludeMaskValues);
                        }

                        // Check Excluded values if necessary
                        if ((renderValue) && (renderer.ExcludedValues.Count>0))
                        {
                            if (renderer.ExcludedValues.Contains(v))
                            {
                                renderValue = false;
                            }
                        }

                        if (renderValue)
                        {
                            // Check range and render values that are within the range
                            if (v >= renderer.MinimumValue && v <= renderer.MaximumValue)
                            {
                                p = renderer.GetPosition(v);
                                if (p >= 0f && p <= 1.0f)
                                {
                                    System.Drawing.Color color = renderer.ColorRamp.GetColor(p);
                                    if (alpha < 255)
                                    { color = Color.FromArgb(alpha, color); }
                                    if (tool.FillBrush != null)
                                    {
                                        brush.Color = color;
                                        if (fillSymbol == null)
                                        {
                                            if (tool.Pen != null)
                                            {
                                                tool.Pen.Color = color;
                                            }
                                        }
                                        else
                                        {
                                            if (fillSymbol.OneColorForFillAndOutline && tool.Pen != null)
                                            { tool.Pen.Color = color; }
                                        }
                                    }
                                    else
                                    {
                                        tool.Pen.Color = color;
                                    }
                                    DrawGeometry(f.Geometry, g, vp, tool, true);
                                }
                            }
                        }
                    }
                }
                catch (Exception)
                {
                    throw;
                }
                finally
                {
                    brush = null;
                    if (tool != null)
                        ((IDisposable)tool).Dispose();
                    if (renderer != null)
                        ((IDisposable)renderer).Dispose();
                }

            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="layer"></param>
        /// <param name="g"></param>
        /// <param name="vp"></param>
        private void RenderClassBreaksRenderer(IFeatureLayer layer, Graphics g, Viewport vp)
        {
            throw new NotImplementedException();
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="geometry"></param>
        /// <param name="g"></param>
        /// <param name="vp"></param>
        /// <param name="tool"></param>
        /// <param name="inflateRectangles"></param>
        private void DrawGeometry(IGeometry geometry, Graphics g, Viewport vp, ISymbolDrawingTool tool, bool inflateRectangles)
        {
            try
            {
                if (geometry is IPolygon)
                {
                    IPolygon pg = geometry as IPolygon;
                    System.Drawing.Brush brush = tool.FillBrush;
                    System.Drawing.Pen pen = null;
                    if (tool.EnableOutline)
                    { pen = tool.Pen; }
                    if (pg.IsRectangle)
                    { vp.DrawEnvelope(pg.EnvelopeInternal, g, pen, brush, inflateRectangles); }
                    else
                    { vp.DrawPolygon(pg, g, pen, brush); }
                }
                else if (geometry is ILineString)
                { vp.DrawLineString(geometry.Coordinates, g, tool.Pen); }
                else if (geometry is IMultiLineString)
                { vp.DrawMultiLineString(geometry as IMultiLineString, g, tool.Pen); }
                else if (geometry is IPoint)
                {
                    IPointSymbol pointSymbol = tool.Symbol as IPointSymbol;
                    IPoint pt = geometry as IPoint;
                    switch (pointSymbol.SymbolType)
                    {
                        case PointSymbolTypes.Circle:
                            vp.DrawCirclePoint(pt, g, tool.Pen, tool.FillBrush, pointSymbol.Size);
                            break;
                        case PointSymbolTypes.Square:
                            vp.DrawSquarePoint(pt, g, tool.Pen, tool.FillBrush, pointSymbol.Size);
                            break;
                        default:
                            break;
                    }

                }
                else
                {
                    // nothing else supported yet.
                }
            }
            catch (Exception)
            {
                // Do not propogate the error, just return.
            }

        }

    }
}
