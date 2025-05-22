using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using GeoAPI.Geometries;
using USGS.Puma.NTS.IO;
using USGS.Puma.NTS.Features;
using USGS.Puma.NTS.Geometries;

namespace USGS.Puma.IO
{
    /// <summary>
    /// Imports and exports ESRI shapefiles
    /// </summary>
    public static class EsriShapefileIO
    {
        /// <summary>
        /// Exports a collection of 3D features to an ESRI shapefile
        /// </summary>
        /// <param name="features"></param>
        /// <param name="directory"></param>
        /// <param name="basename"></param>
        public static void Export(FeatureCollection features, string directory, string basename)
        {
            ShapefileDataWriter writer = null;
            try
            {
                if (features != null)
                {
                    if (features.Count > 0)
                    {
                        string filename = System.IO.Path.Combine(directory, basename);
                        if (ShapefileInfo.TryDelete(directory, basename))
                        {
                            writer = new ShapefileDataWriter(filename);
                            writer.Write(features);
                        }
                        else
                        {
                            throw new Exception("Shapefile '" + filename + "' could not be exported because the existing copy could not be deleted.");
                        }
                    }
                }
            }
            catch (Exception e)
            {
                throw new Exception("Error writing shapefile.", e);
            }
            finally
            {
                // Note: We should call Dispose here, but right now the ShapefileDataWriter
                // class does not support IDisposable. That needs to be fixed. For now, just
                // set writer to null (which should happen anyway, but at least is serves
                // as placeholder code until Dispose is available.
                if (writer != null)
                { 
                    // writer.Dispose();
                    writer = null; 
                }
            }

        }

        /// <summary>
        /// Reads an ESRI shapefile and imports a collection of 3D features.
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public static FeatureCollection Import(string filename)
        {
            FeatureCollection featureList = new FeatureCollection();
            ShapefileDataReader shpDataReader = null;
            try
            {
                shpDataReader = new ShapefileDataReader(filename, new GeometryFactory());
                if (shpDataReader.RecordCount > 0)
                {
                    bool moreRecords;
                    IGeometry geometry = null;
                    string attributeName = null;
                    object attributeValue = null;
                    string[] attributeNames = null;
                    Feature feature = null;
                    AttributesTable attributes = null;

                    int count = 0;
                    while (true)
                    {
                        moreRecords = shpDataReader.Read();
                        if (moreRecords)
                        {
                            count++;
                            geometry = shpDataReader.Geometry;
                            attributes = new AttributesTable();
                            attributeNames = shpDataReader.GetAllNames();
                            for (int i = 0; i < attributeNames.Length; i++)
                            {
                                //Process attributes for all fields except the
                                //Geometry field.
                                if (attributeNames[i] != "Geometry")
                                {
                                    attributeName = attributeNames[i];
                                    attributeValue = shpDataReader[attributeNames[i]];
                                    attributes.AddAttribute(attributeName, attributeValue);
                                }
                            }
                            feature = new Feature(geometry, attributes);
                            featureList.Add(feature);
                        }
                        else
                        { break; }
                    }
                }

            }
            catch (Exception e)
            {
                throw new Exception("Error importing shapefile:" + Environment.NewLine + filename, e);
            }
            finally
            {
                if (shpDataReader != null)
                { shpDataReader.Close(); }
            }

            return featureList;


        }

    }
}
