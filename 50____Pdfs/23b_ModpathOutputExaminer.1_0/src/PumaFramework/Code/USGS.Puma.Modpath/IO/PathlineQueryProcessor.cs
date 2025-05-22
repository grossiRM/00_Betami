using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.Modpath.IO
{
    public static class PathlineQueryProcessor
    {
        #region FilterByID
        /// <summary>
        /// 
        /// </summary>
        /// <param name="list"></param>
        /// <param name="id"></param>
        /// <returns></returns>
        public static IEnumerable<PathlineRecord> FilterByID(IEnumerable<PathlineRecord> list, int id)
        {
            return (list.Where<PathlineRecord>(r => (r.ParticleId == id)));
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="list"></param>
        /// <param name="id"></param>
        /// <returns></returns>
        public static IEnumerable<PathlineRecord> FilterByID(IEnumerable<PathlineRecord> list, int[] id)
        {
            IEnumerable<PathlineRecord> result;
            result = list.Where<PathlineRecord>
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
