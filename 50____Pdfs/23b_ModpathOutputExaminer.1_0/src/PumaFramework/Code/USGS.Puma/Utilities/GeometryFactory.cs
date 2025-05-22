using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.NTS.Features;
using USGS.Puma.NTS.Geometries;
using GeoAPI.Geometries;
using USGS.Puma.Core;

namespace USGS.Puma.Utilities
{
    public class GeometryFactory
    {
        public static USGS.Puma.NTS.Features.Feature[] CreateGridCellPolygons(USGS.Puma.Core.ICellCenteredArealGrid modelGrid, Dictionary<string,object[]> attributeArrays)
        {
            if (modelGrid == null)
                return null;


            USGS.Puma.FiniteDifference.CellCenteredArealGrid grid = null;
            if (modelGrid is USGS.Puma.FiniteDifference.CellCenteredArealGrid)
            {
                grid = modelGrid as USGS.Puma.FiniteDifference.CellCenteredArealGrid;
            }
            else
            {
                throw new NotImplementedException();
            }

            IAttributesTable attributes = null;
            Feature[] features = new Feature[modelGrid.ColumnCount * modelGrid.RowCount];
            ICoordinate[] corners = null;
            ICoordinate[] pts = null;
            Polygon pg = null;
            USGS.Puma.FiniteDifference.GridCell cell = new USGS.Puma.FiniteDifference.GridCell();

            // Compute feature geometry
            int index = -1;
            for (int row = 1; row <= grid.RowCount; row++)
            {
                cell.Row = row;
                for (int column = 1; column <= grid.ColumnCount; column++)
                {
                    index++;
                    cell.Column = column;
                    corners = grid.GetCornerPoints(cell);
                    pts = new ICoordinate[] 
                    {
                        corners[0],
                        corners[1],
                        corners[2],
                        corners[3],
                        new Coordinate(corners[0])
                    };

                    pg = new Polygon(new LinearRing(pts));
                    attributes = new AttributesTable();
                    attributes.AddAttribute("Row", row);
                    attributes.AddAttribute("Column", column);
                    features[index] = new Feature(pg as IGeometry, attributes);

                }
            }

            // Add attributes
            if (attributeArrays != null)
            {
                foreach (KeyValuePair<string, object[]> pair in attributeArrays)
                {
                    if (pair.Value.Length == features.Length)
                    {
                        if ((pair.Key != "Row") && (pair.Key != "Column"))
                        {
                            for (int i = 0; i < features.Length; i++)
                            {
                                features[i].Attributes.AddAttribute(pair.Key, pair.Value[i]);
                            }
                        }
                    }
                }
            }

            return features;

        }
        public static USGS.Puma.NTS.Features.Feature[] CreateGridCellPolygons(USGS.Puma.Core.ICellCenteredArealGrid modelGrid, Dictionary<string, Array2d<float>> attributeArrays)
        {
            if (modelGrid == null)
                return null;


            USGS.Puma.FiniteDifference.CellCenteredArealGrid grid = null;
            if (modelGrid is USGS.Puma.FiniteDifference.CellCenteredArealGrid)
            {
                grid = modelGrid as USGS.Puma.FiniteDifference.CellCenteredArealGrid;
            }
            else
            {
                throw new NotImplementedException();
            }

            IAttributesTable attributes = null;
            Feature[] features = new Feature[modelGrid.ColumnCount * modelGrid.RowCount];
            ICoordinate[] corners = null;
            ICoordinate[] pts = null;
            Polygon pg = null;
            USGS.Puma.FiniteDifference.GridCell cell = new USGS.Puma.FiniteDifference.GridCell();

            // Compute feature geometry
            int index = -1;
            for (int row = 1; row <= grid.RowCount; row++)
            {
                cell.Row = row;
                for (int column = 1; column <= grid.ColumnCount; column++)
                {
                    index++;
                    cell.Column = column;
                    corners = grid.GetCornerPoints(cell);
                    pts = new ICoordinate[] 
                    {
                        corners[0],
                        corners[1],
                        corners[2],
                        corners[3],
                        new Coordinate(corners[0])
                    };
                    pg = new Polygon(new LinearRing(pts));
                    attributes = new AttributesTable();
                    attributes.AddAttribute("Row", row);
                    attributes.AddAttribute("Column", column);
                    features[index] = new Feature(pg as IGeometry, attributes);

                }
            }

            // Add attributes
            if (attributeArrays != null)
            {
                foreach (KeyValuePair<string, Array2d<float>> pair in attributeArrays)
                {
                    if ( (pair.Value.RowCount == grid.RowCount) && (pair.Value.ColumnCount==grid.ColumnCount) )
                    {
                        if ((pair.Key != "Row") && (pair.Key != "Column"))
                        {
                            for (int i = 0; i < features.Length; i++)
                            {
                                int row = Convert.ToInt32(features[i].Attributes["Row"]);
                                int column = Convert.ToInt32(features[i].Attributes["Column"]);
                                features[i].Attributes.AddAttribute(pair.Key, pair.Value[row, column]);
                            }
                        }
                    }
                }
            }

            return features;

        }

        public static USGS.Puma.NTS.Features.Feature[] CreateGridCellNodes(USGS.Puma.Core.ICellCenteredArealGrid modelGrid, Dictionary<string, object[]> attributeArrays)
        {
            if (modelGrid == null)
                return null;


            USGS.Puma.FiniteDifference.CellCenteredArealGrid grid = null;
            if (modelGrid is USGS.Puma.FiniteDifference.CellCenteredArealGrid)
            {
                grid = modelGrid as USGS.Puma.FiniteDifference.CellCenteredArealGrid;
            }
            else
            {
                throw new NotImplementedException();
            }

            IAttributesTable attributes = null;
            Feature[] features = new Feature[modelGrid.ColumnCount * modelGrid.RowCount];
            Point node = null;
            USGS.Puma.FiniteDifference.GridCell cell = new USGS.Puma.FiniteDifference.GridCell();

            // Compute feature geometry
            int index = -1;
            for (int row = 1; row <= grid.RowCount; row++)
            {
                cell.Row = row;
                for (int column = 1; column <= grid.ColumnCount; column++)
                {
                    index++;
                    cell.Column = column;
                    node = new Point(grid.GetNodePoint(cell));
                    attributes = new AttributesTable();
                    attributes.AddAttribute("Row", row);
                    attributes.AddAttribute("Column", column);
                    features[index] = new Feature(node as IGeometry, attributes);
                }
            }

            // Add attributes
            if (attributeArrays != null)
            {
                foreach (KeyValuePair<string, object[]> pair in attributeArrays)
                {
                    if (pair.Value.Length == features.Length)
                    {
                        if ((pair.Key != "Row") && (pair.Key != "Column"))
                        {
                            for (int i = 0; i < features.Length; i++)
                            {
                                features[i].Attributes.AddAttribute(pair.Key, pair.Value[i]);
                            }
                        }
                    }
                }
            }

            return features;

        }
        public static USGS.Puma.NTS.Features.Feature[] CreateGridCellNodes(USGS.Puma.Core.ICellCenteredArealGrid modelGrid, Dictionary<string, Array2d<float>> attributeArrays)
        {
            if (modelGrid == null)
                return null;


            USGS.Puma.FiniteDifference.CellCenteredArealGrid grid = null;
            if (modelGrid is USGS.Puma.FiniteDifference.CellCenteredArealGrid)
            {
                grid = modelGrid as USGS.Puma.FiniteDifference.CellCenteredArealGrid;
            }
            else
            {
                throw new NotImplementedException();
            }

            IAttributesTable attributes = null;
            Feature[] features = new Feature[modelGrid.ColumnCount * modelGrid.RowCount];
            ICoordinate[] corners = null;
            ICoordinate[] pts = null;
            Polygon pg = null;
            Point node = null;
            USGS.Puma.FiniteDifference.GridCell cell = new USGS.Puma.FiniteDifference.GridCell();

            // Compute feature geometry
            int index = -1;
            for (int row = 1; row <= grid.RowCount; row++)
            {
                cell.Row = row;
                for (int column = 1; column <= grid.ColumnCount; column++)
                {
                    index++;
                    cell.Column = column;
                    node = new Point(grid.GetNodePoint(cell));
                    attributes = new AttributesTable();
                    attributes.AddAttribute("Row", row);
                    attributes.AddAttribute("Column", column);
                    features[index] = new Feature(node as IGeometry, attributes);

                }
            }

            // Add attributes
            if (attributeArrays != null)
            {
                foreach (KeyValuePair<string, Array2d<float>> pair in attributeArrays)
                {
                    if ((pair.Value.RowCount == grid.RowCount) && (pair.Value.ColumnCount == grid.ColumnCount))
                    {
                        if ((pair.Key != "Row") && (pair.Key != "Column"))
                        {
                            for (int i = 0; i < features.Length; i++)
                            {
                                int row = Convert.ToInt32(features[i].Attributes["Row"]);
                                int column = Convert.ToInt32(features[i].Attributes["Column"]);
                                features[i].Attributes.AddAttribute(pair.Key, pair.Value[row, column]);
                            }
                        }
                    }
                }
            }

            return features;

        }
        public static USGS.Puma.NTS.Features.Feature[] CreateGridCellNodes(USGS.Puma.Core.ICellCenteredArealGrid modelGrid)
        {
            if (modelGrid == null)
                return null;


            USGS.Puma.FiniteDifference.CellCenteredArealGrid grid = null;
            if (modelGrid is USGS.Puma.FiniteDifference.CellCenteredArealGrid)
            {
                grid = modelGrid as USGS.Puma.FiniteDifference.CellCenteredArealGrid;
            }
            else
            {
                throw new NotImplementedException();
            }

            IAttributesTable attributes = null;
            Feature[] features = new Feature[modelGrid.ColumnCount * modelGrid.RowCount];
            Point node = null;
            USGS.Puma.FiniteDifference.GridCell cell = new USGS.Puma.FiniteDifference.GridCell();

            // Compute feature geometry
            int index = -1;
            for (int row = 1; row <= grid.RowCount; row++)
            {
                cell.Row = row;
                for (int column = 1; column <= grid.ColumnCount; column++)
                {
                    index++;
                    cell.Column = column;
                    node = new Point(grid.GetNodePoint(cell));
                    attributes = new AttributesTable();
                    attributes.AddAttribute("Row", row);
                    attributes.AddAttribute("Column", column);
                    features[index] = new Feature(node as IGeometry, attributes);
                }
            }

            return features;

        }

    }
}
