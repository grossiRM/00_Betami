using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using GeoAPI.Geometries;
using USGS.Puma.NTS.Geometries;

namespace USGS.Puma.Interpolation
{
    public class TriangulatedNetwork
    {
        /// <summary>
        /// Initialize new TriangulatedNetwork
        /// </summary>
        /// <param name="nodeCoordinates"></param>
        /// <param name="triangles"></param>
        public TriangulatedNetwork(ICoordinate[] nodeCoordinates, TriangleNodeConnections[] triangles)
        {
            _NodeCoordinates = nodeCoordinates;
            _Triangles = triangles;
        }

        private TriangleNodeConnections[] _Triangles = null;
        /// <summary>
        /// Gets Triangle node connection
        /// </summary>
        public TriangleNodeConnections[] Triangles
        {
            get { return _Triangles; }
        }

        private ICoordinate[] _NodeCoordinates = null;
        /// <summary>
        /// Gets node coordinate
        /// </summary>
        public ICoordinate[] NodeCoordinates
        {
            get { return _NodeCoordinates; }
            set { _NodeCoordinates = value; }
        }

        public GeometryCollection CreatePolygons()
        {
            throw new NotImplementedException();
        }

       
    }
}
