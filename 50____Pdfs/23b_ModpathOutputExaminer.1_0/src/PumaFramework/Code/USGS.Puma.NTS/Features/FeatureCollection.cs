using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using USGS.Puma;
using USGS.Puma.NTS;
using USGS.Puma.NTS.IO;
using GeoAPI.Geometries;
using USGS.Puma.NTS.Geometries;

namespace USGS.Puma.NTS.Features
{
    /// <summary>
    /// 
    /// </summary>
    public class FeatureCollection : List<Feature>
    {
        #region Public Static Methods
        ///// <summary>
        ///// 
        ///// </summary>
        ///// <param name="filename"></param>
        ///// <returns></returns>
        //public static FeatureCollection ReadShapefile(string filename)
        //{
        //    FeatureCollection featureList = new FeatureCollection();
        //    using (ShapefileDataReader shpDataReader = new ShapefileDataReader(filename, new GeometryFactory()))
        //    {
        //        if (shpDataReader.RecordCount > 0)
        //        {
        //            bool moreRecords;
        //            IGeometry geometry = null;
        //            string attributeName = null;
        //            object attributeValue = null;
        //            string[] attributeNames = null;
        //            Feature feature = null;
        //            AttributesTable attributes = null;

        //            while (true)
        //            {
        //                moreRecords = shpDataReader.Read();
        //                if (moreRecords)
        //                {
        //                    geometry = shpDataReader.Geometry;
        //                    attributes = new AttributesTable();
        //                    attributeNames = shpDataReader.GetAllNames();
        //                    for (int i = 0; i < attributeNames.Length; i++)
        //                    {
        //                        //Process attributes for all fields except the
        //                        //Geometry field.
        //                        if (attributeNames[i] != "Geometry")
        //                        {
        //                            attributeName = attributeNames[i];
        //                            attributeValue = shpDataReader[attributeNames[i]];
        //                            attributes.AddAttribute(attributeName, attributeValue);
        //                        }
        //                    }
        //                    feature = new Feature(geometry, attributes);
        //                    featureList.Add(feature);
        //                }
        //                else
        //                { break; }
        //            }
        //        }
        //    }

        //    return featureList;

        //}

        #endregion

    }
}
