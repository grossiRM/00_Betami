using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.Interpolation
{
	/// <summary>
	/// Triangle made from three point indexes
	/// </summary>
	public struct TriangleNodeConnections
	{
		/// <summary>
		/// First vertex index in triangle
		/// </summary>
		public int Node1;
		/// <summary>
		/// Second vertex index in triangle
		/// </summary>
		public int Node2;
		/// <summary>
		/// Third vertex index in triangle
		/// </summary>
		public int Node3;
        /// <summary>
        /// 
        /// </summary>
        /// <param name="node1"></param>
        /// <param name="node2"></param>
        /// <param name="node3"></param>
		public TriangleNodeConnections(int node1, int node2, int node3)
		{
			Node1 = node1; Node2 = node2; Node3 = node3;
		}
	}
}
