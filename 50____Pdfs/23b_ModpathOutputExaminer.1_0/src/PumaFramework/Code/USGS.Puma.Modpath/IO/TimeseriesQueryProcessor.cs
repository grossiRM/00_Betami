using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.Modpath.IO
{
    public static class TimeseriesQueryProcessor
    {
        #region FilterByTimePoint
        /// <summary>
        /// 
        /// </summary>
        /// <param name="list"></param>
        /// <param name="timePoint"></param>
        /// <returns></returns>
        public static IEnumerable<TimeseriesRecord> FilterByTimePoint(IEnumerable<TimeseriesRecord> list, int timePoint)
        {
            return (list.Where<TimeseriesRecord>(r => (r.TimePoint == timePoint)));
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="list"></param>
        /// <param name="timePoint"></param>
        /// <returns></returns>
        public static IEnumerable<TimeseriesRecord> FilterByTimePoint(IEnumerable<TimeseriesRecord> list, int[] timePoint)
        {
            IEnumerable<TimeseriesRecord> result;
            result = list.Where<TimeseriesRecord>
                (r =>
                {
                    for (int i = 0; i < timePoint.Length; i++)
                    {
                        if ((r.TimePoint == timePoint[i]))
                        { return true; }
                    }
                    return false;
                }
                );
            return (result);

        }

        #endregion

        #region FilterByModflowTimeStep
        /// <summary>
        /// 
        /// </summary>
        /// <param name="list"></param>
        /// <param name="step"></param>
        /// <returns></returns>
        public static IEnumerable<TimeseriesRecord> FilterByModflowTimeStep(IEnumerable<TimeseriesRecord> list, int step)
        {
            return (list.Where<TimeseriesRecord>(r => (r.ModflowTimeStep == step)));
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="list"></param>
        /// <param name="step"></param>
        /// <returns></returns>
        public static IEnumerable<TimeseriesRecord> FilterByModflowTimeStep(IEnumerable<TimeseriesRecord> list, int[] step)
        {
            IEnumerable<TimeseriesRecord> result;
            result = list.Where<TimeseriesRecord>
                (r =>
                {
                    for (int i = 0; i < step.Length; i++)
                    {
                        if ((r.ModflowTimeStep == step[i]))
                        { return true; }
                    }
                    return false;
                }
                );
            return (result);

        }

        #endregion

        #region FilterByID
        /// <summary>
        /// 
        /// </summary>
        /// <param name="list"></param>
        /// <param name="id"></param>
        /// <returns></returns>
        public static IEnumerable<TimeseriesRecord> FilterByID(IEnumerable<TimeseriesRecord> list, int id)
        {
            return (list.Where<TimeseriesRecord>(r => (r.ParticleId == id)));
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="list"></param>
        /// <param name="id"></param>
        /// <returns></returns>
        public static IEnumerable<TimeseriesRecord> FilterByID(IEnumerable<TimeseriesRecord> list, int[] id)
        {
            IEnumerable<TimeseriesRecord> result;
            result = list.Where<TimeseriesRecord>
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

        #endregion

    }
}
