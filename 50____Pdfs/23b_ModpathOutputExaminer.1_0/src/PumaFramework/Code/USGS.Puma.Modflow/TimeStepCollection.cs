using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;

namespace USGS.Puma.Modflow
{
    public class TimeStepCollection : KeyedCollection<string, TimeStep>
    {
        // Wrapper class

        public TimeStepCollection() : base() { }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        protected override string  GetKeyForItem(TimeStep item)
        {
            return item.Key;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="period"></param>
        /// <param name="step"></param>
        /// <returns></returns>
        public bool ContainsTimeStep(int period, int step)
        {
            return this.Contains(TimeStep.CreateKey(period, step));
        }
    }
}
