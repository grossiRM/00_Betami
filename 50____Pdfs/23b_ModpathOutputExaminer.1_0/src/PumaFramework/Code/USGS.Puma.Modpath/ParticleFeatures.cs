using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using GeoAPI.Geometries;
using USGS.Puma.NTS.Geometries;
using USGS.Puma.NTS.Features;
using USGS.Puma.NTS.IO;
using USGS.Puma.Modflow;
using USGS.Puma.Modpath.IO;

namespace USGS.Puma.Modpath
{
    public static class ParticleFeatures
    {
        #region Public Methods
        /// <summary>
        /// Creates an endpoint features collection from a list of endpoint records
        /// </summary>
        /// <param name="endpointRecords"></param>
        /// <param name="endpointOption"></param>
        /// <returns></returns>
        public static FeatureCollection CreateEndpointFeatures(List<EndpointRecord> endpointRecords, EndpointLocationTypes endpointOption)
        {
            return CreateEndpointFeatures(endpointRecords, endpointOption, false);
        }
        /// <summary>
        /// Creates an endpoint features collection from a list of endpoint records
        /// </summary>
        /// <param name="endpointRecords"></param>
        /// <param name="endpointOption"></param>
        /// <returns></returns>
        public static FeatureCollection CreateEndpointFeatures(List<EndpointRecord> endpointRecords, EndpointLocationTypes endpointOption, bool particleIdOnly)
        {
            ICoordinate c;
            USGS.Puma.NTS.Geometries.Point pt;
            FeatureCollection features = new FeatureCollection();
            Feature feature;
            AttributesTable attrib;
            EndpointRecord rec;

            for (int i = 0; i < endpointRecords.Count; i++)
            {
                rec = endpointRecords[i];
                if (endpointOption == EndpointLocationTypes.InitialPoint)
                {
                    c = (ICoordinate)(new Coordinate(Convert.ToDouble(rec.InitialX), Convert.ToDouble(rec.InitialY), Convert.ToDouble(rec.InitialZ)));
                }
                else
                {
                    c = (ICoordinate)(new Coordinate(Convert.ToDouble(rec.FinalX), Convert.ToDouble(rec.FinalY), Convert.ToDouble(rec.FinalZ)));
                }
                pt = new USGS.Puma.NTS.Geometries.Point(c);
                attrib = new AttributesTable();
                attrib.AddAttribute("ParticleId", rec.ParticleId);
                if (!particleIdOnly)
                {
                    attrib.AddAttribute("Group", rec.Group);
                    attrib.AddAttribute("Status", rec.Status);
                    attrib.AddAttribute("InitZone", rec.InitialZone);
                    attrib.AddAttribute("FinalZone", rec.FinalZone);
                    attrib.AddAttribute("TravelTime", (rec.FinalTime - rec.InitialTime));
                    attrib.AddAttribute("InitGrid", rec.InitialGrid);
                    attrib.AddAttribute("InitLayer", rec.InitialLayer);
                    attrib.AddAttribute("InitRow", rec.InitialRow);
                    attrib.AddAttribute("InitCol", rec.InitialColumn);
                    attrib.AddAttribute("FinalGrid", rec.FinalGrid);
                    attrib.AddAttribute("FinalLayer", rec.FinalLayer);
                    attrib.AddAttribute("FinalRow", rec.FinalRow);
                    attrib.AddAttribute("FinalCol", rec.FinalColumn);
                }
                feature = new Feature(pt as IGeometry, attrib);
                features.Add(feature);

            }

            return features;

        }

        /// <summary>
        /// Creates a timeseries feature collection from a list of timeseries records
        /// </summary>
        /// <param name="timeseriesRecords"></param>
        /// <returns></returns>
        public static FeatureCollection CreateTimeseriesFeatures(List<TimeseriesRecord> timeseriesRecords)
        {
            return CreateTimeseriesFeatures(timeseriesRecords, null, false);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="timeseriesRecords"></param>
        /// <param name="particleIdOnly"></param>
        /// <returns></returns>
        public static FeatureCollection CreateTimeseriesFeatures(List<TimeseriesRecord> timeseriesRecords, bool particleIdOnly)
        {
            return CreateTimeseriesFeatures(timeseriesRecords, null, particleIdOnly);
        }
        /// <summary>
        /// Creates a timeseries features collection from a list of timeseries records
        /// </summary>
        /// <param name="list"></param>
        /// <returns></returns>
        public static FeatureCollection CreateTimeseriesFeatures(List<TimeseriesRecord> timeseriesRecords, List<EndpointRecord> endpointRecords, bool particleIdOnly)
        {
            ICoordinate c;
            USGS.Puma.NTS.Geometries.Point pt;
            FeatureCollection features = new FeatureCollection();
            Feature feature;
            AttributesTable attrib;
            TimeseriesRecord rec;
            int currentParticleId = -1;
            EndpointRecord[] epr = null;
            
            for (int i = 0; i < timeseriesRecords.Count; i++)
            {
                rec = timeseriesRecords[i];
                c = (ICoordinate)(new Coordinate(Convert.ToDouble(rec.X), Convert.ToDouble(rec.Y), Convert.ToDouble(rec.Z)));
                pt = new USGS.Puma.NTS.Geometries.Point(c);
                attrib = new AttributesTable();
                attrib.AddAttribute("ParticleId", rec.ParticleId);
                if (!particleIdOnly)
                {
                    attrib.AddAttribute("Group", rec.Group);
                    attrib.AddAttribute("TimePoint", rec.TimePoint);
                    attrib.AddAttribute("TimeStep", rec.ModflowTimeStep);
                    attrib.AddAttribute("Time", rec.Time);
                    attrib.AddAttribute("Elevation", rec.Z);
                    attrib.AddAttribute("Layer", rec.Layer);
                    if (endpointRecords != null)
                    {
                        if (currentParticleId != rec.ParticleId)
                        {
                            currentParticleId = rec.ParticleId;
                            epr = EndpointQueryProcessor.FilterByID(endpointRecords, currentParticleId).ToArray<EndpointRecord>();
                            if ((epr == null) || (epr.Length == 0))
                            {
                                throw new Exception("Timeseries and endpoint record lists are inconsistent.");
                            }
                        }
                        attrib.AddAttribute("InitZone", epr[0].InitialZone);
                        attrib.AddAttribute("FinalZone", epr[0].FinalZone);
                        attrib.AddAttribute("TravelTime", (rec.Time - epr[0].InitialTime));
                    }
                }
                feature = new Feature(pt as IGeometry, attrib);
                features.Add(feature);
            }

            return features;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="pathlineRecords"></param>
        /// <param name="mergePathlineParts"></param>
        /// <returns></returns>
        public static FeatureCollection CreatePathlineFeatures(List<PathlineRecord> pathlineRecords, bool mergePathlineParts)
        {
            return CreatePathlineFeatures(pathlineRecords, null, mergePathlineParts, false);
        }
        /// <summary>
        /// Creates a pathline feature collection from a list of pathline records
        /// </summary>
        /// <param name="pathlineRecords"></param>
        /// <param name="mergePathlineParts"></param>
        /// <returns></returns>
        public static FeatureCollection CreatePathlineFeatures(List<PathlineRecord> pathlineRecords, List<EndpointRecord> endpointRecords, bool mergePathlineParts)
        {
            return CreatePathlineFeatures(pathlineRecords, endpointRecords, mergePathlineParts, false);
        }
        /// <summary>
        /// Creates a pathline features collection from a list of pathline records
        /// </summary>
        /// <param name="list"></param>
        /// <param name="mergePathlineParts"></param>
        /// <returns></returns>
        public static FeatureCollection CreatePathlineFeatures(List<PathlineRecord> pathlineRecords, List<EndpointRecord> endpointRecords, bool mergePathlineParts, bool particleIdOnly)
        {
            if (pathlineRecords == null)
            { throw new ArgumentNullException(); }

            bool createSegment = false;
            IGeometry geom = null;
            ICoordinate c = null;
            double time0 = 0.0;
            PathlineRecord prevRec = null;
            AttributesTable attr = null;
            FeatureCollection features = new FeatureCollection();
            List<int> idValues = new List<int>();
            List<ICoordinate> coords = new List<ICoordinate>();

            // Find all unique id values
            foreach (PathlineRecord rec in pathlineRecords)
            {
                if (!idValues.Contains(rec.ParticleId))
                { idValues.Add(rec.ParticleId); }
            }
            int[] id = idValues.ToArray<int>();

            EndpointRecord epr = null;

            double releaseTime = 0.0;
            double travelTime = 0.0;

            // Loop through each particle
            for (int i = 0; i < id.Length; i++)
            {
                // Get the records for the current particle ID value.
                PathlineRecord[] recs = PathlineQueryProcessor.FilterByID(pathlineRecords, id[i]).ToArray<PathlineRecord>();
                if (endpointRecords != null)
                {
                    EndpointRecord[] epRecs = EndpointQueryProcessor.FilterByID(endpointRecords, id[i]).ToArray<EndpointRecord>();
                    if ((epRecs == null) || (epRecs.Length == 0))
                    {
                        throw new Exception("The enpoint and pathline record lists are not consistent.");
                    }
                    epr = epRecs[0];
                }
                // Loop through the records for the current particle
                coords.Clear();
                prevRec = null;
                for (int n = 0; n < recs.Length; n++)
                {
                    // Set the release time value if this is the first point
                    if (n == 0)
                    { releaseTime = recs[0].Time; }

                    // Check for and skip duplicate locations.
                    if (!IsDuplicatePathlineLocation(prevRec, recs[n]))
                    {
                        // Compute and add a new coordinate point
                        c = new Coordinate(Convert.ToDouble(recs[n].X), Convert.ToDouble(recs[n].Y), Convert.ToDouble(recs[n].Z));
                        (c as ICoordinateM).M = Convert.ToDouble(recs[n].Time);
                        coords.Add(c);

                        // Determine whether the accumulated coordinates should be
                        // used to create a new pathline segment.
                        if (coords.Count == 1)
                        {
                            time0 = Convert.ToDouble(recs[n].Time);
                            createSegment = false;
                        }
                        else
                        {
                            if (n == recs.Length - 1)
                            { createSegment = true; }
                            else
                            {
                                if (mergePathlineParts)
                                { createSegment = false; }
                                else
                                {
                                    if (recs[n].TimePoint > 0) createSegment = true;
                                }
                            }
                        }

                        // Create a new pathline segment and add it to features collection
                        if (createSegment)
                        {
                            geom = (CreatePathline(coords.ToArray<ICoordinate>())) as IGeometry;
                            attr = new AttributesTable();
                            attr.AddAttribute("ParticleId", recs[n].ParticleId);
                            if (!particleIdOnly)
                            {
                                attr.AddAttribute("Group", recs[n].Group);
                                attr.AddAttribute("Time0", time0);
                                attr.AddAttribute("Time", recs[n].Time);
                                if (endpointRecords != null)
                                {
                                    travelTime = epr.FinalTime - epr.InitialTime;
                                    attr.AddAttribute("TravelTime", travelTime);
                                    attr.AddAttribute("InitZone", epr.InitialZone);
                                    attr.AddAttribute("FinalZone", epr.FinalZone);
                                }
                            }
                            features.Add(new Feature(geom, attr));
                            coords.Clear();
                            prevRec = null;
                        }
                        else
                        {
                            prevRec = recs[n];
                        }

                    }
                    else
                    {
                        prevRec = recs[n];
                        // Flush the last pathline segment if this is the final 
                        // record and the coordinate list is not empty.
                        if (n == recs.Length - 1)
                        {
                            if (coords.Count > 1)
                            {
                                geom = (CreatePathline(coords.ToArray<ICoordinate>())) as IGeometry;
                                attr = new AttributesTable();
                                attr.AddAttribute("ParticleId", recs[n - 1].ParticleId);
                                if (!particleIdOnly)
                                {
                                    attr.AddAttribute("Group", recs[n - 1].Group);
                                    attr.AddAttribute("Time0", time0);
                                    attr.AddAttribute("Time", recs[n - 1].Time);
                                    if (endpointRecords != null)
                                    {
                                        travelTime = epr.FinalTime - epr.InitialTime;
                                        attr.AddAttribute("TravelTime", travelTime);
                                        attr.AddAttribute("InitZone", epr.InitialZone);
                                        attr.AddAttribute("FinalZone", epr.FinalZone);
                                    }
                                }
                                features.Add(new Feature(geom, attr));
                                coords.Clear();
                                prevRec = null;
                            }
                        }
                    }
                }
            }

            return features;

        }
        
        #endregion

        #region Private Methods
        private static IMultiLineString CreatePathline(ICoordinate[] coords)
        {
            ILineString ls = new LineString(coords);
            ILineString[] lsArray = new ILineString[1] { ls };
            IMultiLineString mls = new MultiLineString(lsArray);
            return mls;
        }
        private static bool IsDuplicatePathlineLocation(PathlineRecord rec1, PathlineRecord rec2)
        {
            if (rec1 == null) return false;
            if (rec2 == null) return false;

            if (rec1.ParticleId != rec2.ParticleId) return false;
            if ((rec1.X != rec2.X) || (rec1.Y != rec2.Y) || (rec1.Z != rec2.Z)) return false;

            return true;

        }
        #endregion
    }
}
