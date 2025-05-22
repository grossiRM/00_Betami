using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.FiniteDifference
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public class GridCellReach : GridCell
    {
        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="GridCellReach"/> class.
        /// </summary>
        /// <remarks></remarks>
        public GridCellReach() : base()
        {
            Group = 0;
            ExtraLength = 0.0;
            ReachLength = 0.0;
            Position = 0.0;
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="GridCellReach"/> class.
        /// </summary>
        /// <param name="grid">The grid.</param>
        /// <param name="layer">The layer.</param>
        /// <param name="row">The row.</param>
        /// <param name="column">The column.</param>
        /// <param name="coordinates">The coordinates.</param>
        /// <remarks></remarks>
        public GridCellReach(int group, int grid, int layer, int row, int column, double reachLength, double extraLength, double position)
        {
            Group = group;
            Grid = grid;
            Layer = layer;
            Row = row;
            Column = column;
            ReachLength = reachLength;
            ExtraLength = extraLength;
            Position = position;
        }
        #endregion

        #region Public Properties
        private int _Group;
        public int Group
        {
            get { return _Group; }
            set { _Group = value; }
        }
        /// <summary>
        /// 
        /// </summary>
        private double _ExtraLength;
        /// <summary>
        /// Gets or sets the extra length that will be added to the reach length to obtain the total length.
        /// </summary>
        /// <value>The length of the reach.</value>
        /// <remarks></remarks>
        public double ExtraLength
        {
            get { return _ExtraLength; }
            set { _ExtraLength = value; }
        }
        /// <summary>
        /// Gets the total length.
        /// </summary>
        /// <remarks></remarks>
        public double TotalLength
        {
            get
            {
                return ExtraLength + ReachLength;
            }
        }
        private double _ReachLength;
        public double ReachLength
        {
            get { return _ReachLength; }
            set { _ReachLength = value; }
        }
        private double _Position;
        public double Position
        {
            get { return _Position; }
            set { _Position = value; }
        }
        /// <summary>
        /// 
        /// </summary>
        private USGS.Puma.NTS.Features.IAttributesTable _Attributes = null;
        /// <summary>
        /// Gets or sets the attributes.
        /// </summary>
        /// <value>The attributes.</value>
        /// <remarks></remarks>
        public USGS.Puma.NTS.Features.IAttributesTable Attributes
        {
            get { return _Attributes; }
            set { _Attributes = value; }
        }
        #endregion

    }
}
