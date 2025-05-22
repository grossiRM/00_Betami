using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.Modpath.IO
{
    public static class EndpointQueryProcessor
    {
        #region FilterByID
        /// <summary>
        /// Returns endpoint records where the particle ID matches the argument id.
        /// </summary>
        /// <param name="list"></param>
        /// <param name="id"></param>
        /// <returns></returns>
        public static IEnumerable<EndpointRecord> FilterByID(IEnumerable<EndpointRecord> list, int id)
        {
            return (list.Where<EndpointRecord>(r => (r.ParticleId == id)));
        }
        /// <summary>
        /// Returns endpoint records where the particle ID matches any of the values in the input array, id[].
        /// </summary>
        /// <param name="list"></param>
        /// <param name="id"></param>
        /// <returns></returns>
        public static IEnumerable<EndpointRecord> FilterByID(IEnumerable<EndpointRecord> list, int[] id)
        {
            IEnumerable<EndpointRecord> result;
            result = list.Where<EndpointRecord>
                (r =>
                    {
                        for (int i = 0; i < id.Length; i++)
                        {
                            if ((r.ParticleId == id[i]))
                            { return true; }
                        }
                        return false;
                    }
                );
            return (result);
        }
        //public static IEnumerable<EndpointRecord> FilterByID(IEnumerable<EndpointRecord> list, int[] id)
        //{
        //    IEnumerable<EndpointRecord> result = null;
        //    for (int i = 0; i < id.Length; i++)
        //    {
        //        if (i == 0)
        //        {
        //            result = FilterByID(list, id[0]);
        //        }
        //        else
        //        {
        //            if (result != null)
        //            {
        //                result = result.Union<EndpointRecord>(FilterByID(list, id[i]));
        //            }
        //        }
        //    }
        //    return (result);
        //}
        #endregion

        #region FilterByGroup
        /// <summary>
        /// Returns endpoint records where the particle Group matches the argument group.
        /// </summary>
        /// <param name="list"></param>
        /// <param name="id"></param>
        /// <returns></returns>
        public static IEnumerable<EndpointRecord> FilterByGroup(IEnumerable<EndpointRecord> list, int group)
        {
            return (list.Where<EndpointRecord>(r => (r.Group == group)));
        }
        /// <summary>
        /// Returns endpoint records where the particle Group matches any of the values in the input array, group[].
        /// </summary>
        /// <param name="list"></param>
        /// <param name="id"></param>
        /// <returns></returns>
        public static IEnumerable<EndpointRecord> FilterByGroup(IEnumerable<EndpointRecord> list, int[] group)
        {
            IEnumerable<EndpointRecord> result;
            result = list.Where<EndpointRecord>
                (r =>
                    {
                        for (int i = 0; i < group.Length; i++)
                        {
                            if ((r.Group == group[i]))
                            { return true; }
                        }
                        return false;
                    }
                );
            return (result);
        }
        #endregion

        #region FilterByStatus
        /// <summary>
        /// 
        /// </summary>
        /// <param name="list"></param>
        /// <param name="status"></param>
        /// <returns></returns>
        public static IEnumerable<EndpointRecord> FilterByStatus(IEnumerable<EndpointRecord> list, int status)
        {
            return (list.Where<EndpointRecord>(r => (r.Status == status)));

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="list"></param>
        /// <param name="statusFlagCheck"></param>
        /// <returns></returns>
        public static IEnumerable<EndpointRecord> FilterByStatus(IEnumerable<EndpointRecord> list, bool[] statusFlagCheck)
        {
            IEnumerable<EndpointRecord> result;
            result = list.Where<EndpointRecord>
                (r =>
                    {
                        for (int i = 0; i < statusFlagCheck.Length; i++)
                        {
                            if (i > 5)
                            { return false; }
                            if (statusFlagCheck[i])
                            {
                                if ((r.Status == i))
                                { return true; }
                            }
                        }
                        return false;
                    }
                );

            return (result);

        }
        #endregion

        #region FilterByTravelTime
        /// <summary>
        /// Returns endpoint records that match the specified travel time criterion.
        /// </summary>
        /// <param name="list"></param>
        /// <param name="travelTime"></param>
        /// <param name="equalityOption"></param>
        /// <returns></returns>
        public static IEnumerable<EndpointRecord> FilterByTravelTime(IEnumerable<EndpointRecord> list, float travelTime, QueryEquality equalityOption)
        {
            switch (equalityOption)
            {
                case QueryEquality.LessThan:
                    return (list.Where<EndpointRecord>(r => (r.FinalTime - r.InitialTime) < travelTime));

                case QueryEquality.LessThanOrEqual:
                    return (list.Where<EndpointRecord>(r => (r.FinalTime - r.InitialTime) <= travelTime));

                case QueryEquality.Equal:
                    return (list.Where<EndpointRecord>(r => (r.FinalTime - r.InitialTime) == travelTime));

                case QueryEquality.GreaterThanOrEqual:
                    return (list.Where<EndpointRecord>(r => (r.FinalTime - r.InitialTime) >= travelTime));

                case QueryEquality.GreaterThan:
                    return (list.Where<EndpointRecord>(r => (r.FinalTime - r.InitialTime) > travelTime));

                default:
                    throw new ArgumentException("Invalid QueryEquality option");
            }
        }
        #endregion

        #region FilterByTime
        public static IEnumerable<EndpointRecord> FilterByTime(IEnumerable<EndpointRecord> list, float time, EndpointLocationTypes endpoint, QueryEquality equalityOption)
        {
            switch (endpoint)
            {
                case EndpointLocationTypes.InitialPoint:
                    switch (equalityOption)
                    {
                        case QueryEquality.LessThan:
                            return (list.Where<EndpointRecord>(r => r.InitialTime < time));

                        case QueryEquality.LessThanOrEqual:
                            return (list.Where<EndpointRecord>(r => r.InitialTime <= time));

                        case QueryEquality.Equal:
                            return (list.Where<EndpointRecord>(r => r.InitialTime == time));

                        case QueryEquality.GreaterThanOrEqual:
                            return (list.Where<EndpointRecord>(r => r.InitialTime >= time));

                        case QueryEquality.GreaterThan:
                            return (list.Where<EndpointRecord>(r => r.InitialTime > time));

                        default:
                            throw new ArgumentException("Invalid QueryEquality option");
                    }

                case EndpointLocationTypes.FinalPoint:
                    switch (equalityOption)
                    {
                        case QueryEquality.LessThan:
                            return (list.Where<EndpointRecord>(r => r.FinalTime < time));

                        case QueryEquality.LessThanOrEqual:
                            return (list.Where<EndpointRecord>(r => r.FinalTime <= time));

                        case QueryEquality.Equal:
                            return (list.Where<EndpointRecord>(r => r.FinalTime == time));

                        case QueryEquality.GreaterThanOrEqual:
                            return (list.Where<EndpointRecord>(r => r.FinalTime >= time));

                        case QueryEquality.GreaterThan:
                            return (list.Where<EndpointRecord>(r => r.FinalTime > time));

                        default:
                            throw new ArgumentException("Invalid QueryEquality option");
                    }

                default:
                    throw new ArgumentException("Invalid QueryEndpoint option.");
            }
        }

        #endregion

        #region FilterByGrid
        /// <summary>
        /// Returns endpoint records where the Grid value of the specified endpoint matches the argument grid.
        /// </summary>
        /// <param name="list"></param>
        /// <param name="id"></param>
        /// <returns></returns>
        public static IEnumerable<EndpointRecord> FilterByGrid(IEnumerable<EndpointRecord> list, int grid, EndpointLocationTypes endpoint)
        {
            switch (endpoint)
            {
                case EndpointLocationTypes.InitialPoint:
                    return (list.Where<EndpointRecord>(r => (r.InitialGrid == grid)));
                   
                case EndpointLocationTypes.FinalPoint:
                    return (list.Where<EndpointRecord>(r => (r.FinalGrid == grid)));
                   
                default:
                    throw new ArgumentException("Invalid QueryEndpoint option.");
            }
        }
        /// <summary>
        /// Returns endpoint records where the Grid value of the specified endpoint matches any of the 
        /// values in the input array, grid[].
        /// </summary>
        /// <param name="list"></param>
        /// <param name="id"></param>
        /// <returns></returns>
        public static IEnumerable<EndpointRecord> FilterByGrid(IEnumerable<EndpointRecord> list, int[] grid, EndpointLocationTypes endpoint)
        {
            IEnumerable<EndpointRecord> result;
            switch (endpoint)
            {
                case EndpointLocationTypes.InitialPoint:
                    result = list.Where<EndpointRecord>
                        (r =>
                            {
                                for (int i = 0; i < grid.Length; i++)
                                {
                                    if ((r.InitialGrid == grid[i]))
                                    { return true; }
                                }
                                return false;
                            }
                        );
                    return (result);
                    
                case EndpointLocationTypes.FinalPoint:
                    result = list.Where<EndpointRecord>
                       (r =>
                           {
                               for (int i = 0; i < grid.Length; i++)
                               {
                                   if ((r.FinalGrid == grid[i]))
                                   { return true; }
                               }
                               return false;
                           }
                       );
                    return (result);
                   
                default:
                    throw new ArgumentException("Invalid QueryEndpoint option.");
            }

        }
        #endregion

        #region FilterByCell
        /// <summary>
        /// Returns endpoint records where the cell indices of the specified endpoint match the input layer, 
        /// row, and column.
        /// </summary>
        /// <param name="list"></param>
        /// <param name="layer"></param>
        /// <param name="row"></param>
        /// <param name="column"></param>
        /// <param name="endpoint"></param>
        /// <returns></returns>
        public static IEnumerable<EndpointRecord> FilterByCell(IEnumerable<EndpointRecord> list, int layer, int row, int column, EndpointLocationTypes endpoint)
        {
            switch (endpoint)
            {
                case EndpointLocationTypes.InitialPoint:
                    return (list.Where<EndpointRecord>(r => (r.InitialLayer == layer) && (r.InitialRow == row) && (r.InitialColumn == column)));

                case EndpointLocationTypes.FinalPoint:
                    return (list.Where<EndpointRecord>(r => (r.FinalLayer == layer) && (r.FinalRow == row) && (r.FinalColumn == column)));

                default:
                    throw new ArgumentException("Invalid QueryEndpoint option.");
            }
        }
        /// <summary>
        /// Returns endpoint records where the cell indices of the specified endpoint match the input layer,
        /// row, and column.
        /// </summary>
        /// <param name="list"></param>
        /// <param name="layer"></param>
        /// <param name="row"></param>
        /// <param name="column"></param>
        /// <param name="endpoint"></param>
        /// <returns></returns>
        public static IEnumerable<EndpointRecord> FilterByCell(IEnumerable<EndpointRecord> list, int[] layer, int[] row, int[] column, EndpointLocationTypes endpoint)
        {
            IEnumerable<EndpointRecord> result;
            switch (endpoint)
            {
                case EndpointLocationTypes.InitialPoint:
                    result = list.Where<EndpointRecord>
                       (r =>
                           {
                               for (int i = 0; i < layer.Length; i++)
                               {
                                   if ((r.InitialLayer == layer[i]) && (r.InitialRow == row[i]) && (r.InitialColumn == column[i]))
                                   { return true; }
                               }
                               return false;
                           }
                       );
                    return (result);

                case EndpointLocationTypes.FinalPoint:
                    result = list.Where<EndpointRecord>
                        (r =>
                            {
                                for (int i = 0; i < layer.Length; i++)
                                {
                                    if ((r.FinalLayer == layer[i]) && (r.FinalRow == row[i]) && (r.FinalColumn == column[i]))
                                    { return true; }
                                }
                                return false;
                            }
                        );
                    return (result);

                default:
                    throw new ArgumentException("Invalid QueryEndpoint option.");
            }
        }
        #endregion

        #region FilterByLayer
        public static IEnumerable<EndpointRecord> FilterByLayer(IEnumerable<EndpointRecord> list, int layer, EndpointLocationTypes endpoint)
        {
            switch (endpoint)
            {
                case EndpointLocationTypes.InitialPoint:
                    return (list.Where<EndpointRecord>(r => (r.InitialLayer == layer)));

                case EndpointLocationTypes.FinalPoint:
                    return (list.Where<EndpointRecord>(r => (r.FinalLayer == layer)));

                default:
                    throw new ArgumentException("Invalid QueryEndpoint option.");
            }
        }
        #endregion

        #region FilterByRow
        public static IEnumerable<EndpointRecord> FilterByRow(IEnumerable<EndpointRecord> list, int row, EndpointLocationTypes endpoint)
        {
            switch (endpoint)
            {
                case EndpointLocationTypes.InitialPoint:
                    return (list.Where<EndpointRecord>(r => (r.InitialRow == row)));

                case EndpointLocationTypes.FinalPoint:
                    return (list.Where<EndpointRecord>(r => (r.FinalRow == row)));

                default:
                    throw new ArgumentException("Invalid QueryEndpoint option.");
            }
        }
        #endregion

        #region FilterByColumn
        public static IEnumerable<EndpointRecord> FilterByColumn(IEnumerable<EndpointRecord> list, int column, EndpointLocationTypes endpoint)
        {
            switch (endpoint)
            {
                case EndpointLocationTypes.InitialPoint:
                    return (list.Where<EndpointRecord>(r => (r.InitialColumn == column)));

                case EndpointLocationTypes.FinalPoint:
                    return (list.Where<EndpointRecord>(r => (r.FinalColumn == column)));

                default:
                    throw new ArgumentException("Invalid QueryEndpoint option.");
            }
        }
        #endregion

        #region FilterByFace
        public static IEnumerable<EndpointRecord> FilterByFace(IEnumerable<EndpointRecord> list, int face, EndpointLocationTypes endpoint)
        {
            switch (endpoint)
            {
                case EndpointLocationTypes.InitialPoint:
                    return (list.Where<EndpointRecord>(r => (r.InitialFace == face)));

                case EndpointLocationTypes.FinalPoint:
                    return (list.Where<EndpointRecord>(r => (r.FinalFace == face)));

                default:
                    throw new ArgumentException("Invalid QueryEndpoint option.");
            }
        }
        #endregion

        #region FilterByZone
        public static IEnumerable<EndpointRecord> FilterByZone(IEnumerable<EndpointRecord> list, int zone, EndpointLocationTypes endpoint)
        {
            switch (endpoint)
            {
                case EndpointLocationTypes.InitialPoint:
                    return (list.Where<EndpointRecord>(r => (r.InitialZone == zone)));

                case EndpointLocationTypes.FinalPoint:
                    return (list.Where<EndpointRecord>(r => (r.FinalZone == zone)));

                default:
                    throw new ArgumentException("Invalid QueryEndpoint option.");
            }
        }

        #endregion

        #region LocalX
        public static IEnumerable<EndpointRecord> FilterByLocalX(IEnumerable<EndpointRecord> list, int localX, EndpointLocationTypes endpoint)
        {
            switch (endpoint)
            {
                case EndpointLocationTypes.InitialPoint:
                    return (list.Where<EndpointRecord>(r => (r.InitialLocalX == localX)));

                case EndpointLocationTypes.FinalPoint:
                    return (list.Where<EndpointRecord>(r => (r.FinalLocalX == localX)));

                default:
                    throw new ArgumentException("Invalid QueryEndpoint option.");
            }
        }

        #endregion

        #region LocalY
        public static IEnumerable<EndpointRecord> FilterByLocalY(IEnumerable<EndpointRecord> list, int localY, EndpointLocationTypes endpoint)
        {
            switch (endpoint)
            {
                case EndpointLocationTypes.InitialPoint:
                    return (list.Where<EndpointRecord>(r => (r.InitialLocalY == localY)));

                case EndpointLocationTypes.FinalPoint:
                    return (list.Where<EndpointRecord>(r => (r.FinalLocalY == localY)));

                default:
                    throw new ArgumentException("Invalid QueryEndpoint option.");
            }
        }

        #endregion

        #region LocalZ
        public static IEnumerable<EndpointRecord> FilterByLocalZ(IEnumerable<EndpointRecord> list, int localZ, EndpointLocationTypes endpoint)
        {
            switch (endpoint)
            {
                case EndpointLocationTypes.InitialPoint:
                    return (list.Where<EndpointRecord>(r => (r.InitialLocalZ == localZ)));

                case EndpointLocationTypes.FinalPoint:
                    return (list.Where<EndpointRecord>(r => (r.FinalLocalZ == localZ)));

                default:
                    throw new ArgumentException("Invalid QueryEndpoint option.");
            }
        }

        #endregion

        #region X
        public static IEnumerable<EndpointRecord> FilterByX(IEnumerable<EndpointRecord> list, int x, EndpointLocationTypes endpoint)
        {
            switch (endpoint)
            {
                case EndpointLocationTypes.InitialPoint:
                    return (list.Where<EndpointRecord>(r => (r.InitialX == x)));

                case EndpointLocationTypes.FinalPoint:
                    return (list.Where<EndpointRecord>(r => (r.FinalX == x)));

                default:
                    throw new ArgumentException("Invalid QueryEndpoint option.");
            }
        }

        #endregion

        #region Y
        public static IEnumerable<EndpointRecord> FilterByY(IEnumerable<EndpointRecord> list, int y, EndpointLocationTypes endpoint)
        {
            switch (endpoint)
            {
                case EndpointLocationTypes.InitialPoint:
                    return (list.Where<EndpointRecord>(r => (r.InitialY == y)));

                case EndpointLocationTypes.FinalPoint:
                    return (list.Where<EndpointRecord>(r => (r.FinalY == y)));

                default:
                    throw new ArgumentException("Invalid QueryEndpoint option.");
            }
        }
        #endregion

        #region Z
        public static IEnumerable<EndpointRecord> FilterByZ(IEnumerable<EndpointRecord> list, int z, EndpointLocationTypes endpoint)
        {
            switch (endpoint)
            {
                case EndpointLocationTypes.InitialPoint:
                    return (list.Where<EndpointRecord>(r => (r.InitialZ == z)));

                case EndpointLocationTypes.FinalPoint:
                    return (list.Where<EndpointRecord>(r => (r.FinalZ == z)));

                default:
                    throw new ArgumentException("Invalid QueryEndpoint option.");
            }
        }

        #endregion

        /// <summary>
        /// Gets a list of distinct initial time values.
        /// </summary>
        /// <param name="list"></param>
        /// <returns>A list of initial time values.</returns>
        public static List<float> GetInitialTimeValues(IEnumerable<EndpointRecord> list)
        {
            List<float> times = list.Select(r => r.InitialTime).Distinct().ToList<float>();
            times.Sort();
            return times;
        }
    }
}
