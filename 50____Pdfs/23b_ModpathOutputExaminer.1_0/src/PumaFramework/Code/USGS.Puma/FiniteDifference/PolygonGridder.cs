using System;
using System.Collections.Generic;
using System.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using USGS.Puma.NTS.Geometries;
using USGS.Puma.NTS.Features;
using GeoAPI.Geometries;

namespace USGS.Puma.FiniteDifference
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public class PolygonGridder
    {
        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="T:System.Object"/> class.
        /// </summary>
        /// <remarks></remarks>
        public PolygonGridder()
        {
            // default constructor
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="PolygonGridder"/> class.
        /// </summary>
        /// <param name="grid">The grid.</param>
        /// <remarks></remarks>
        public PolygonGridder(CellCenteredArealGrid grid)
        {
            Grid = grid;
        }
        #endregion

        #region Private Fields
        /// <summary>
        /// 
        /// </summary>
        private double[] _NodeX = null;
        /// <summary>
        /// 
        /// </summary>
        private double[] _NodeY = null;
        #endregion

        #region Public Properties
        /// <summary>
        /// 
        /// </summary>
        private CellCenteredArealGrid _Grid = null;
        /// <summary>
        /// Gets or sets the grid.
        /// </summary>
        /// <value>The grid.</value>
        /// <remarks></remarks>
        public CellCenteredArealGrid Grid
        {
            get { return _Grid; }
            set 
            { 
                _Grid = value;

                // Create arrays to hold the x and y node locations in relative coordinates and cache them so that
                // they can be reused in each gridding operation on this model grid.
                _NodeX = new double[_Grid.ColumnCount + 1];
                _NodeY = new double[_Grid.RowCount + 1];

                _NodeX[0] = 0.0;
                _NodeX[1] = _Grid.GetColumnSpacing(1) / 2.0;
                for (int column = 2; column <= _Grid.ColumnCount; column++)
                {
                    _NodeX[column] = _NodeX[column - 1] + ((_Grid.GetColumnSpacing(column-1) + _Grid.GetColumnSpacing(column)) / 2.0);
                }

                _NodeY[0] = 0.0;
                _NodeY[1] = _Grid.TotalRowHeight - (_Grid.GetRowSpacing(1) / 2.0);
                for (int row = 2; row <= _Grid.RowCount; row++)
                {
                    _NodeY[row] = _NodeY[row - 1] - ((_Grid.GetRowSpacing(row - 1) + _Grid.GetRowSpacing(row)) / 2.0);
                }

            }
        }

        #endregion

        #region Public Methods
        /// <summary>
        /// Generates the gridded output from an input polygon.
        /// </summary>
        /// <param name="polygon">The polygon.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public GridCellList ExecuteGridding(IPolygon polygon)
        {
            if (polygon == null)
            { throw new ArgumentNullException("polygon"); }

            IPolygon poly = null;
            Point pt = new Point(0.0, 0.0);
            GridCellList list = new GridCellList();

            GridCellRegion gridCellEnvelope = GetGridCellEnvelope(polygon);
            if (gridCellEnvelope != null)
            {
                bool rotatePoints = false;
                if (Grid.OriginX != 0.0 || Grid.OriginY != 0.0 || Grid.Angle != 0.0) rotatePoints = true;
                if (rotatePoints)
                {
                    // Create a copy of the polygon so that we won't actually be rotating the points in the 
                    // original polygon. This costs a little extra up front, but it saves a little on the
                    // back end because there is no need to rotate the points back to the original locations
                    // at the end.
                    poly = polygon.Clone() as IPolygon;

                    // The method polygon.Coordinates returns a new array each time it is referenced that contains references to all the
                    // coordinates internal to the polygon. Assign polygon.Coordinates once at the beginning to variable coords
                    // to avoid creating a new array everytime pass through the coordinates loop.
                    ICoordinate[] coords = poly.Coordinates;
                    for (int i = 0; i < coords.Length; i++)
                    {
                        Grid.TransformGlobalToRelative(coords[i]);
                    }
                }
                else
                {
                    poly = polygon;
                }

                for (int row = gridCellEnvelope.FromCell.Row; row <= gridCellEnvelope.ToCell.Row; row++)
                {
                    for (int column = gridCellEnvelope.FromCell.Column; column <= gridCellEnvelope.ToCell.Column; column++)
                    {
                        pt.X = _NodeX[column];
                        pt.Y = _NodeY[row];
                        if (poly.Contains(pt))
                        {
                            list.Add(new GridCell(row, column));
                        }
                    }
                }

            }

            return list;

        }
        /// <summary>
        /// Generates the gridded output from an input polygon feature.
        /// </summary>
        /// <param name="polygonFeature">The polygon feature.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public GridCellList ExecuteGridding(Feature polygonFeature)
        {
            if (polygonFeature == null)
            {
                throw new ArgumentNullException("polygonFeature");
            }
            if(!(polygonFeature.Geometry is IPolygon))
            {
                throw new ArgumentException("The specified feature geometry is not a polygon.");
            }

            IPolygon polygon = polygonFeature.Geometry as IPolygon;
            return ExecuteGridding(polygon);
            
        }
        /// <summary>
        /// Gets an instance of the <see cef="GridCellRegion"/> class that defines the grid node points contained within the specified <see cref="IEnvlope"/> region.
        /// </summary>
        /// <param name="envelope">The envelope.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public GridCellRegion GetGridCellEnvelope(IEnvelope envelope)
        {
            return GetGridCellEnvelope(envelope.MinX, envelope.MaxX, envelope.MinY, envelope.MaxY);
        }
        /// <summary>
        /// Gets an instance of the <see cef="GridCellRegion"/> class that defines the grid node points contained within the specified row and column limits.
        /// </summary>
        /// <param name="minX">The min X.</param>
        /// <param name="maxX">The max X.</param>
        /// <param name="minY">The min Y.</param>
        /// <param name="maxY">The max Y.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public GridCellRegion GetGridCellEnvelope(double minimumX, double maximumX, double minimumY, double maximumY)
        {
            double minX = minimumX;
            double maxX = maximumX;
            double minY = minimumY;
            double maxY = maximumY;
            Coordinate c = new Coordinate();

            bool rotatePoints = false;
            if (Grid.OriginX != 0.0 || Grid.OriginY != 0.0 || Grid.Angle != 0.0) rotatePoints = true;
            if (rotatePoints)
            {
                c.X = minX;
                c.Y = minY;
                Grid.TransformGlobalToRelative(c);
                minX = c.X;
                minY = c.Y;
                c.X = maxX;
                c.Y = maxY;
                Grid.TransformGlobalToRelative(c);
                maxX = c.X;
                maxY = c.Y;
            }

            if (Grid == null)
            {
                throw new Exception("The model grid has not been set.");
            }

            // Find minimum column
            int minColumn = 0;
            for (int column = 1; column <= Grid.ColumnCount; column++)
            {
                if (_NodeX[column] >= minX)
                {
                    minColumn = column;
                    break;
                }
            }
            if (minColumn == 0) return null;

            // Find maximum column
            int maxColumn = 0;
            for (int column = Grid.ColumnCount; column >= 1; column--)
            {
                if (_NodeX[column] <= maxX)
                {
                    maxColumn = column;
                    break;
                }
            }
            if (maxColumn < minColumn) return null;

            // Find minimum row
            int minRow = 0;
            for (int row = 1; row <= Grid.RowCount; row++)
            {
                if (_NodeY[row] <= maxY)
                {
                    minRow = row;
                    break;
                }
            }
            if (minRow == 0) return null;

            // Find maximum row
            int maxRow = 0;
            for (int row = Grid.RowCount; row >= 1; row--)
            {
                if (_NodeY[row] <= minY)
                {
                    maxRow = row;
                    break;
                }
            }
            if (maxRow < minRow) return null;

            // Create and return a grid cell region
            GridCell fromCell = new GridCell(minRow, minColumn);
            GridCell toCell = new GridCell(maxRow, maxColumn);
            return new GridCellRegion(fromCell, toCell);

        }
        /// <summary>
        /// Gets an instance of the <see cef="GridCellRegion"/> class that defines the grid node points contained within envelope of the specified <see cref="IPolygon"/> region.
        /// </summary>
        /// <param name="polygon">The polygon.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public GridCellRegion GetGridCellEnvelope(IPolygon polygon)
        {
            IEnvelope env = polygon.Envelope as IEnvelope;
            return GetGridCellEnvelope(env);
        }
        #endregion

        #region Private Methods
        #endregion

    }
}
