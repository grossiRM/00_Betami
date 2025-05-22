using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Xml;

namespace USGS.Puma.Modflow
{
    public class XmlLayerDataWriter<T>
    {
        public static void Write(BinaryLayerReader reader, string filename)
        {
            Write(reader, filename, "LayerDataRecordList");
        }
        public static void Write(BinaryLayerReader reader, string filename, string elementName)
        {
            if (reader == null)
                throw new ArgumentNullException("The specified binary file reader does not exist.");

            // Make sure the file gets deleted if it already exists. This
            // should not be necessary, but sometimes the file is not
            // updated properly if it is a preexisting file.
            System.IO.FileInfo fileInfo = new System.IO.FileInfo(filename);
            if (fileInfo.Exists) fileInfo.Delete();

            using (StreamWriter writer = new StreamWriter(filename))
            {
                using (XmlTextWriter xmlWriter = new XmlTextWriter(writer))
                {
                    xmlWriter.Formatting = System.Xml.Formatting.Indented;
                    xmlWriter.Indentation = 2;

                    // Write xml document declaration
                    xmlWriter.WriteStartDocument();

                    // Write the starting tag for the root element
                    xmlWriter.WriteStartElement(elementName);

                    for (int i = 0; i < reader.RecordCount; i++)
                    {
                        if (reader.OutputPrecision == OutputPrecisionType.Single)
                        {
                            LayerDataRecord<float> rec = reader.GetRecordAsSingle(i);
                            rec.Write(xmlWriter);
                        }
                        else if (reader.OutputPrecision == OutputPrecisionType.Double)
                        {
                            LayerDataRecord<double> rec = reader.GetRecordAsDouble(i);
                            rec.Write(xmlWriter);
                        }
                    }

                    // Close the root element
                    xmlWriter.WriteEndElement();

                    xmlWriter.Close();
                }
                writer.Close();
            }

        }
    }
}
