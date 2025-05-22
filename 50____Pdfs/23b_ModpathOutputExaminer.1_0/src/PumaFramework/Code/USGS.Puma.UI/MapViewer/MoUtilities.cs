using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace USGS.Puma.UI.MapViewer
{
    public class MoUtilities
    {
        public static UInt32 ToMoColor(System.Drawing.Color color)
        {
            byte[] b = new byte[4] { color.R, color.G, color.B, 0 };
            return Convert.ToUInt32(BitConverter.ToInt32(b, 0));
        }
        public static UInt32 ToMoColor(int red, int green, int blue)
        {
            byte[] b = new byte[4] { Convert.ToByte(red), Convert.ToByte(green), Convert.ToByte(blue), 0 };
            return Convert.ToUInt32(BitConverter.ToInt32(b, 0));
        }

        public static System.Drawing.Color FromMoColor(UInt32 color)
        {
            byte[] b = BitConverter.GetBytes(color);
            int red = Convert.ToInt32(b[0]);
            int green = Convert.ToInt32(b[1]);
            int blue = Convert.ToInt32(b[2]);
            return System.Drawing.Color.FromArgb(red, green, blue);
        }

        public static void CopyShapefile(string shapefilename, string destination)
        {
            try
            {
                if (!Directory.Exists(destination))
                    return;

                if (!Path.HasExtension(shapefilename))
                {
                    shapefilename = shapefilename + ".shp";
                }

                if (Path.GetExtension(shapefilename).ToLower() == ".shp")
                {
                    string dir = Path.GetDirectoryName(shapefilename);
                    string basename = Path.GetFileNameWithoutExtension(shapefilename);
                    string basePathname = Path.Combine(dir, basename);

                    // Check to see if the three necessary files exist.
                    bool valid = true;
                    if (!File.Exists(shapefilename))
                        valid = false;
                    if (!File.Exists(basePathname + ".shx"))
                        valid = false;
                    if (!File.Exists(basePathname + ".dbf"))
                        valid = false;

                    if (valid)
                    {
                        string sourceFile = shapefilename;
                        string destinationFile = Path.Combine(destination, basename + ".shp");
                        File.Copy(sourceFile, destinationFile);

                        destinationFile = Path.Combine(destination, basename + ".shx");
                        File.Copy(basePathname + ".shx", destinationFile);

                        destinationFile = Path.Combine(destination, basename + ".dbf");
                        File.Copy(basePathname + ".dbf", destinationFile);

                        if (File.Exists(basePathname + ".prj"))
                        {
                            destinationFile = Path.Combine(destination, basename + ".prj");
                            File.Copy(basePathname + ".prj", destinationFile);
                        }

                        if (File.Exists(basePathname+".sbn"))
                        {
                            destinationFile = Path.Combine(destination, basename + ".sbn");
                            File.Copy(basePathname + ".sbn", destinationFile);
                        }

                    }

                }

            }
            catch (Exception)
            {
                throw;
            }

        }

    }
}
