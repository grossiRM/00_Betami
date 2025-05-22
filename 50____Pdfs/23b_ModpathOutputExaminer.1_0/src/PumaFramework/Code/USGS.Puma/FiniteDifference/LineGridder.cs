using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using USGS.Puma.NTS.Geometries;
using GeoAPI.Geometries;

namespace USGS.Puma.FiniteDifference
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public class LineGridder
    {
        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="T:System.Object"/> class.
        /// </summary>
        /// <remarks></remarks>
        public LineGridder()
        {
            // default public constructor
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="LineGridder"/> class.
        /// </summary>
        /// <param name="grid">The grid.</param>
        /// <remarks></remarks>
        public LineGridder(CellCenteredArealGrid grid)
        {
            Grid = grid;
        }
        #endregion

        #region Public Properties
        /// <summary>
        /// 
        /// </summary>
        private CellCenteredArealGrid _Grid;
        /// <summary>
        /// Gets or sets the areal finite-difference grid.
        /// </summary>
        /// <value>The grid.</value>
        /// <remarks></remarks>
        public CellCenteredArealGrid Grid
        {
            get { return _Grid; }
            set { _Grid = value; }
        }
        #endregion

        #region Public Methods
        /// <summary>
        /// Splits the line at grid cell boundaries.
        /// </summary>
        /// <param name="coordinates">The coordinates.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public ICoordinateM[] SplitLineAtCellBoundaries(ICoordinate[] coordinates)
        {
            if (coordinates == null)
            { throw new ArgumentNullException("coodinates"); }

            if (coordinates.Length < 2)
            { throw new ArgumentException("The coordinates array must contain at least two elements."); }

            // Create copies of the line coordinates and set the measure property (M) equal to the 
            // cummulative 2D distance along the line. This is all necessary to assure that we are working 
            // with copies of the coordinates so that any changes will not affect the original line.
            // It also allows us to work with points that are transformed back to an unrotated grid
            // if the working grid is rotated and/or has an origin other than (0,0).
            ICoordinateM[] sourceLine = new ICoordinateM[coordinates.Length];
            List<ICoordinateM> splitLine = new List<ICoordinateM>();

            double dist = 0.0;

            bool rotatePoints = false;
            if (Grid.OriginX != 0.0 || Grid.OriginY != 0.0 || Grid.Angle != 0.0) rotatePoints = true;
            
            for (int i = 0; i < sourceLine.Length; i++)
            {
                // Make a copy of point i
                sourceLine[i] = coordinates[i].Clone() as ICoordinateM;
                
                // Transform from global to relative coordinates if necessary
                if (rotatePoints)
                {
                    Grid.TransformGlobalToRelative(sourceLine[i]);
                }

                // Calculate cummulative distance for point i and store it in the measure property, M
                if (i > 0)
                {
                    // The Distance method of the Coordinate class computes the distance in the x-y plane.
                    // It ignores the Z coordinate.
                    dist +=  sourceLine[i].Distance(sourceLine[i - 1]);
                }
                sourceLine[i].M = dist;

                // Add a copy of the point to splitLine
                splitLine.Add(sourceLine[i].Clone() as ICoordinateM);
            }

            // We now have two copies of the original coordinates that have M set equal to the cumulative areal distance.
            // They also have been transformed to a relative coordinate system, if necessary. Now we can make two passes
            // through the grid to find and the intersection points for the column grid lines and the row grid lines and
            // then add them to splitLine. Because the splitLine is a List object, new points can be inserted easily.

            // Loop through all line segments
            for (int i = 1; i < sourceLine.Length; i++)
            {
                ICoordinateM fromPoint = sourceLine[i - 1];
                ICoordinateM toPoint = sourceLine[i];

                double segDx = toPoint.X - fromPoint.X;
                double segDy = toPoint.Y - fromPoint.Y;

                // Pass 1: Find intersections with column grid lines. 
                #region Pass 1
                // Skip this block of code if the line segment is vertical.
                if (fromPoint.X != toPoint.X)
                {
                    if (toPoint.X > fromPoint.X)  
                    {
                        // Line segment points in the positive x direction
                        double columnBoundary = 0.0;
                        for (int column = 0; column <= Grid.ColumnCount; column++)
                        {
                            if (column > 0)
                            { columnBoundary += Grid.GetColumnSpacing(column); }
                            if (column == Grid.ColumnCount)
                            { columnBoundary = Grid.TotalColumnWidth; }

                            if (fromPoint.X < columnBoundary)
                            {
                                // Exit the column loop if the line segment does not extend
                                // beyond the current column boundary because there are
                                // no more intersection points to be found.
                                if (toPoint.X <= columnBoundary) break;

                                // Find the intersection point with the current column boundary.
                                double dx = columnBoundary - fromPoint.X;
                                double dy = dx * segDy / segDx;  // calculate dy based on similar triangles
                                double y = fromPoint.Y + dy;

                                // If the point falls within the model grid, add it to the splitLine coordinate list.
                                if (y >= 0.0 && y <= Grid.TotalRowHeight)
                                {
                                    ICoordinateM pt = new Coordinate(fromPoint.X + dx, y);
                                    pt.M = fromPoint.M + fromPoint.Distance(pt);
                                    int index = FindInsertIndexFromM(splitLine, pt);
                                    // If index has a positive value the point should be added to the splitLine list.
                                    if (index > 0)
                                    {
                                        splitLine.Insert(index, pt);
                                    }
                                }
                            }
                        }
                    }
                    else  
                    {
                        // Line segment points in the negative x direction
                        double columnBoundary = Grid.TotalColumnWidth;
                        for (int column = Grid.ColumnCount + 1; column >= 1; column--)
                        {
                            if (column <= Grid.ColumnCount)
                            { columnBoundary -= Grid.GetColumnSpacing(column); }
                            if (column == 1)
                            { columnBoundary = 0.0; }

                            if (fromPoint.X > columnBoundary)
                            {
                                // Exit the column loop if the line segment does not extend
                                // beyond the current column boundary because there are
                                // no more intersection points to be found.
                                if (columnBoundary <= toPoint.X) break;

                                // Find the intersection point with the current column boundary.
                                double dx = columnBoundary - fromPoint.X;
                                double dy = dx * segDy / segDx;
                                double y = fromPoint.Y + dy;

                                // If the point falls within the model grid, add it to the splitLine coordinate list.
                                if (y >= 0.0 && y <= Grid.TotalRowHeight)
                                {
                                    ICoordinateM pt = new Coordinate(fromPoint.X + dx, y);
                                    pt.M = fromPoint.M + fromPoint.Distance(pt);
                                    int index = FindInsertIndexFromM(splitLine, pt);
                                    // If index has a positive value the point should be added to the splitLine list.
                                    if (index >= 0)
                                    {
                                        splitLine.Insert(index, pt);
                                    }
                                }
                            }

                        }
                    }
                }
                #endregion

                // Pass 2: Find intersections with row grid lines.
                #region Pass 2
                // Skip this block if the line segment is horizontal
                if (fromPoint.Y != toPoint.Y)
                {
                    // Line segment points in the positive y direction
                    if (toPoint.Y > fromPoint.Y)
                    {
                        double rowBoundary = 0.0;
                        for (int row = Grid.RowCount+1; row >= 1; row--)
                        {
                            if (row <= Grid.RowCount)
                            { rowBoundary += Grid.GetRowSpacing(row); }
                            if (row == 1)
                            { rowBoundary = Grid.TotalRowHeight; }

                            if (fromPoint.Y < rowBoundary)
                            {
                                if (toPoint.Y <= rowBoundary) break;

                                double dy = rowBoundary - fromPoint.Y;
                                double dx = dy * segDx / segDy;
                                double x = fromPoint.X + dx;
                                if (x >= 0.0 && x <= Grid.TotalColumnWidth)
                                {
                                    ICoordinateM pt = new Coordinate(x, fromPoint.Y + dy);
                                    pt.M = fromPoint.M + fromPoint.Distance(pt);
                                    int index = FindInsertIndexFromM(splitLine, pt);
                                    if (index >= 0)
                                    {
                                        splitLine.Insert(index, pt);
                                    }
                                }
                            }
                        }

                    }
                    else
                    {
                        // Line segment points in the negative y direction
                        double rowBoundary = Grid.TotalRowHeight;
                        for (int row = 0; row <= Grid.RowCount; row++)
                        {
                            if (row > 0)
                            { rowBoundary -= Grid.GetRowSpacing(row); }
                            if (row == Grid.RowCount)
                            { rowBoundary = 0.0; }

                            if (row == Grid.RowCount)
                            {
                                rowBoundary = 0.0;
                            }
                            if (fromPoint.Y > rowBoundary)
                            {
                                if (toPoint.Y >= rowBoundary) break;

                                double dy = rowBoundary - fromPoint.Y;
                                double dx = dy * segDx / segDy;
                                double x = fromPoint.X + dx;

                                if (x >= 0.0 && x <= Grid.TotalColumnWidth)
                                {
                                    ICoordinateM pt = new Coordinate(x, fromPoint.Y + dy);
                                    pt.M = fromPoint.M + fromPoint.Distance(pt);
                                    int index = FindInsertIndexFromM(splitLine, pt);
                                    if (index >= 0)
                                    {
                                        splitLine.Insert(index, pt);
                                    }
                                }
                            }
                        }
                    }
                }
                #endregion
            }

            // return the splitLine list as an array of coordinates
            if (rotatePoints)
            {
                for (int i = 0; i < splitLine.Count; i++)
                {
                    Grid.TransformRelativeToGlobal(splitLine[i]);
                }
            }
            return splitLine.ToArray();

        }
        /// <summary>
        /// Splits the line at cell boundaries.
        /// </summary>
        /// <param name="line">The line.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public ICoordinateM[] SplitLineAtCellBoundaries(ILineString line)
        {
            if (line == null)
            { throw new ArgumentNullException("line"); }

            // Retrieve the coordinates of the linestring object
            ICoordinateM[] coords = line.Coordinates as ICoordinateM[];
            return SplitLineAtCellBoundaries(coords);
        }
        /// <summary>
        /// Creates the gridded reach list from split line.
        /// </summary>
        /// <param name="splitLine">The split line.</param>
        /// <param name="tolerance">The tolerance.</param>
        /// <param name="group">The group.</param>
        /// <param name="gridID">The grid ID.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public GridCellReachList CreateReachListFromSplitLine(ICoordinateM[] splitLine, double tolerance, int group, int gridID)
        {
            if (Grid == null)
            {
                throw new Exception("The model grid is not defined.");
            }

            if (splitLine == null)
            {
                throw new ArgumentNullException("splitLine");
            }

            if (splitLine.Length < 2)
            {
                throw new ArgumentException("The specified coordinate array must contain at least two points.");
            }

            GridCellReachList list = new GridCellReachList();
            GridCellReach reach = null;
            double cellSize = 0.0;
            double extraLength = 0.0;
            int startIndex = -1;
            GridCell cell = null;
            Coordinate c = new Coordinate();
            GridCell targetCell = null;

            // Loop through the line segments
            for (int i = 1; i < splitLine.Length; i++)
            {
                // Compute line segment midpoint.
                c.X = (splitLine[i - 1].X + splitLine[i].X) / 2.0;
                c.Y = (splitLine[i - 1].Y + splitLine[i].Y) / 2.0;
                // Find out which grid cell contains the midpoint. If the midpoint is outside the grid, return null.
                cell = Grid.FindRowColumn(c);

                if (cell != null)  
                {
                    // The line segment is within the model grid

                    if (targetCell == null)
                    {
                        startIndex = i - 1;
                        targetCell = new GridCell(cell.Row, cell.Column);
                        extraLength = 0.0;
                        // If this is the last line segment, wrap up any loose ends and exit
                        if (i == splitLine.Length - 1)
                        {
                            reach = CreateReachInternal(targetCell, splitLine, startIndex, i - 1, gridID, group);
                            if (ReachLengthMeetsTolerance(reach, tolerance))
                            {
                                reach.ExtraLength = extraLength;
                                extraLength = 0.0;
                                list.Add(reach);
                            }
                            break;
                        }
                    }
                    else
                    {
                        // A target cell exists, so check to see if cell and targetCell point to the same grid cell.
                        // If so, continue. Otherwise, create a GridCellReach object for targetCell,
                        // add it to the list, and then prepare for the next line segment.
                        if ((cell.Row != targetCell.Row) || (cell.Column != targetCell.Column))
                        {
                            reach = CreateReachInternal(targetCell, splitLine, startIndex, i - 1, gridID, group);
                            if (ReachLengthMeetsTolerance(reach, tolerance))
                            {
                                reach.ExtraLength = extraLength;
                                extraLength = 0.0;
                                list.Add(reach);
                            }
                            else
                            {
                                extraLength += reach.ReachLength;

                                // If this is the last line segment, wrap up any loose ends and exit
                                if (i == splitLine.Length - 1)
                                {
                                    if (list.Count > 0)
                                    {
                                        list[list.Count - 1].ExtraLength += extraLength;
                                    }
                                    break;
                                }
                            }

                            // Prepare for the next line segment
                            targetCell = new GridCell(cell.Row, cell.Column);
                            startIndex = i - 1;

                            // If this is the last line segment, clean up loose ends and exit
                            if (i == splitLine.Length - 1)
                            {
                                reach = CreateReachInternal(targetCell, splitLine, startIndex, i, gridID, group);
                                if (ReachLengthMeetsTolerance(reach, tolerance))
                                {
                                    reach.ExtraLength = extraLength;
                                    extraLength = 0.0;
                                    list.Add(reach);
                                }
                                else
                                {
                                    extraLength += reach.ReachLength;
                                    list[list.Count - 1].ExtraLength += extraLength;
                                    break;
                                }

                            }

                        }
                        else
                        {
                            // If this is the last line segment, clean up loose ends and exit
                            if (i == splitLine.Length - 1)
                            {
                                reach = CreateReachInternal(targetCell, splitLine, startIndex, i, gridID, group);
                                if (ReachLengthMeetsTolerance(reach, tolerance))
                                {
                                    reach.ExtraLength = extraLength;
                                    extraLength = 0.0;
                                    list.Add(reach);
                                }
                                else
                                {
                                    extraLength += reach.ReachLength;
                                    list[list.Count - 1].ExtraLength += extraLength;
                                    break;
                                }
                            }
                        }
                    }
                }
                else  
                {
                    // The line segment is outside the model grid.

                    // If there is reach pending for a target cell, create and process the reach, then prepare for the next line segment.
                    if (targetCell != null)
                    {
                        reach = CreateReachInternal(targetCell, splitLine, startIndex, i - 1, gridID, group);
                        if (ReachLengthMeetsTolerance(reach, tolerance))
                        {
                            reach.ExtraLength = extraLength;
                            extraLength = 0.0;
                            list.Add(reach);
                        }
                        else
                        {
                            extraLength += reach.ReachLength;

                            // If this is the last line segment, wrap up any loose ends and exit
                            if (i == splitLine.Length - 1)
                            {
                                if (list.Count > 0)
                                {
                                    list[list.Count - 1].ExtraLength += extraLength;
                                }
                                break;
                            }
                        }

                        // The current line segment is outside the grid, so set targetCell to null before going to the next line segment.
                        targetCell = null;
                        extraLength = 0.0;

                    }
                }
            }

            return list;
        }
        /// <summary>
        /// Splits the line and create gridded reach list.
        /// </summary>
        /// <param name="sourceLine">The source line.</param>
        /// <param name="tolerance">The tolerance.</param>
        /// <param name="group">The group.</param>
        /// <param name="gridID">The grid ID.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public GridCellReachList CreateReachListFromFeature(USGS.Puma.NTS.Features.Feature sourceLine, double tolerance,int group,int gridID)
        {
            GridCellReachList list = null;
            if (sourceLine is IMultiLineString)
            {
                IMultiLineString line = sourceLine.Geometry as IMultiLineString;
                if (line.NumGeometries == 1)
                {
                    ICoordinateM[] splitLine = SplitLineAtCellBoundaries(sourceLine.Geometry.Coordinates);
                    list = CreateReachListFromSplitLine(splitLine, tolerance, group, gridID);
                }
                else
                {
                    throw new ArgumentException("Multipart line features are not allowed.");
                }
            }
            else
            {
                throw new ArgumentException("The specified feature is not the correct geometry type.", "sourceLine");
            }
            return list;
        }
        /// <summary>
        /// Creates the reach lists from features.
        /// </summary>
        /// <param name="features">The features.</param>
        /// <param name="tolerance">The tolerance.</param>
        /// <param name="group">The group.</param>
        /// <param name="gridID">The grid ID.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public GridCellReachList[] CreateReachListsFromFeatures(USGS.Puma.NTS.Features.FeatureCollection features, double tolerance, int group, int gridID)
        {
            GridCellReachList[] list = new GridCellReachList[features.Count];
            for (int i = 0; i < features.Count; i++)
            {
                list[i] = CreateReachListFromFeature(features[i], tolerance, group, gridID);
            }
            return list;
        }
        #endregion

        #region Private Methods
        /// <summary>
        /// Finds the insert index from M value of the specified coordinate.
        /// </summary>
        /// <param name="coordinates">The coordinates.</param>
        /// <param name="point">The point.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private int FindInsertIndexFromM(List<ICoordinateM> coordinates, ICoordinateM point)
        {
            if (point == null || coordinates == null)
            { return -1; }

            if (coordinates.Count < 2)
            { return -1; }

            if (point.M > 0.0 && point.M < coordinates[coordinates.Count - 1].M)
            {
                for (int i = 1; i < coordinates.Count; i++)
                {
                    if (point.M > coordinates[i - 1].M && point.M < coordinates[i].M)
                    {
                        return i;
                    }
                }
                return -1;
            }
            else
            {
                return -1;
            }


        }
        /// <summary>
        /// Creates the reach internal.
        /// </summary>
        /// <param name="targetCell">The target cell.</param>
        /// <param name="splitLine">The split line.</param>
        /// <param name="startIndex">The start index.</param>
        /// <param name="endIndex">The end index.</param>
        /// <param name="gridID">The grid ID.</param>
        /// <param name="group">The group.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private GridCellReach CreateReachInternal(GridCell targetCell, ICoordinateM[] splitLine, int startIndex, int endIndex, int gridID, int group)
        {
            double reachLength = splitLine[endIndex].M - splitLine[startIndex].M;
            double position = (splitLine[startIndex].M + splitLine[endIndex].M) / 2.0 / splitLine[splitLine.Length - 1].M;
            GridCellReach reach = new GridCellReach(group, gridID, targetCell.Layer, targetCell.Row, targetCell.Column, reachLength, 0.0, position);
            return reach;
        }
        /// <summary>
        /// Reaches the length meets tolerance.
        /// </summary>
        /// <param name="reach">The reach.</param>
        /// <param name="tolerance">The tolerance.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private bool ReachLengthMeetsTolerance(GridCellReach reach, double tolerance)
        {
            double width = Grid.GetColumnSpacing(reach.Column);
            double height = Grid.GetRowSpacing(reach.Row);
            double cellsize = width;
            if (height < cellsize) cellsize = height;
            double ratio = reach.ReachLength / cellsize;
            if (ratio < tolerance)
            {
                return false;
            }
            else
            {
                return true;
            }
        }

        #endregion
    }
}
