using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Core;

namespace USGS.Puma.Utilities
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public class StringUtility
    {
        /// <summary>
        /// Chars the array to string.
        /// </summary>
        /// <param name="ca">The ca.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public static string CharArrayToString(char[] ca)
        {
            if (ca == null) return "";

            string[] sa = new string[ca.Length];

            for (int i = 0; i < ca.Length; i++)
            {
                sa[i] = ca[i].ToString();
            }

            return string.Concat(sa);

        }

        /// <summary>
        /// Trims the extra spaces.
        /// </summary>
        /// <param name="sourceLine">The source line.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public static string TrimExtraSpaces(string sourceLine)
        {
            string s = null;

            try
            {
                if (sourceLine == null) return "";

                s = sourceLine.Trim();
                string doubleSpace = "  ";
                string singleSpace = " ";

                do
                { s = s.Replace(doubleSpace, singleSpace); }
                while (s.IndexOf(doubleSpace) >= 0);

                return s;

            }
            catch (Exception ex)
            {
                throw;
            }
            return null;
        }

        /// <summary>
        /// Splits the fixed line.
        /// </summary>
        /// <param name="sourceLine">The source line.</param>
        /// <param name="fieldWidth">Width of the field.</param>
        /// <param name="fieldCount">The field count.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public static string[] SplitFixedLine(string sourceLine, int fieldWidth, int fieldCount)
        {
            //  This function returns a string array from a fixed format line.
            //  It's supposed to work like the split command.
            List<string> list = new List<string>();
            int start = 0;

            for (int i = 0; i < fieldCount; i++)
            {
                start = (i * fieldWidth);

                // Check remaining string and return if the remaining length < fieldWidth
                int endPosition = start + fieldWidth;
                if (endPosition > sourceLine.Length) return list.ToArray();

                list.Add(sourceLine.Substring(start, fieldWidth));
            }

            return list.ToArray();

        }

        /// <summary>
        /// Gets the next token from a StreamReader stream. Tokens are separated by spaces or
        /// carriage return, or new line characters.
        /// </summary>
        /// <param name="reader">The reader.</param>
        /// <param name="sb">The sb.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public static string GetNextToken(System.IO.StreamReader reader, System.Text.StringBuilder sb)
        {
            if (reader == null)
                throw new NullReferenceException();
            if (sb == null)
                throw new NullReferenceException();

            char c;
            char sp = ' ';
            char comma = ',';
            char cr = System.Convert.ToChar("\r");
            char nl = System.Convert.ToChar("\n");

            sb.Length = 0;
            bool haveFirst = false;

            while (reader.Peek() >= 0)
            {
                c = Convert.ToChar(reader.Read());
                if (c == sp || c == comma || c == cr || c == nl)
                {
                    if (haveFirst)
                    {
                        if (sb.Length > 0)
                            return sb.ToString(0, sb.Length);
                        else
                            return "";
                    }
                }
                else
                {
                    haveFirst = true;
                    sb.Append(c);
                }
            }

            // End of file was reached so return what we have.
            if (sb.Length > 0)
                return sb.ToString(0, sb.Length);
            else
                return "";

        }

        /// <summary>
        /// Parses as fortran free format.
        /// </summary>
        /// <param name="textLine">The text line.</param>
        /// <param name="useDoubleQuote">if set to <c>true</c> [use double quote].</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public static List<string> ParseAsFortranFreeFormat(string textLine, bool useDoubleQuote)
        {
            try
            {
                List<string> tokens = new List<string>();
                StringBuilder sb = new StringBuilder(250);

                string sLine = textLine.Trim();
                if (!string.IsNullOrEmpty(sLine))
                {
                    Char comma = ',';
                    Char space = ' ';

                    string squote = "'";
                    Char quote = squote[0];
                    if (useDoubleQuote) quote = '"';

                    int mode = 1;
                    bool usingQuotes = false;
                    for (int i = 0; i < sLine.Length; i++)
                    {
                        Char c = sLine[i];
                        if (mode == 1)
                        {
                            if (c == quote)
                            {
                                mode = 2;
                                sb.Length = 0;
                                usingQuotes = true;
                            }
                            else
                            {
                                if ((c != comma) && (c != space))
                                {
                                    mode = 2;
                                    usingQuotes = false;
                                    sb.Length = 0;
                                    sb.Append(c);
                                }
                            }
                        }
                        else
                        {
                            if (usingQuotes)
                            {
                                if ((c == quote) || (i + 1 == sLine.Length))
                                {
                                    tokens.Add(sb.ToString(0, sb.Length));
                                    sb.Length = 0;
                                    mode = 1;
                                    usingQuotes = false;
                                }
                                else
                                {
                                    sb.Append(c);
                                }
                            }
                            else
                            {
                                int ii = i + 1;
                                if (ii == sLine.Length)
                                {
                                    sb.Append(c);
                                    tokens.Add(sb.ToString(0, sb.Length));
                                    sb.Length = 0;
                                }
                                else
                                {
                                    if ((c == comma) || (c == space))
                                    {
                                        tokens.Add(sb.ToString(0, sb.Length));
                                        sb.Length = 0;
                                        mode = 1;
                                        usingQuotes = false;
                                    }
                                    else
                                    {
                                        sb.Append(c);
                                    }
                                }
                            }
                        }
                    }
                }

                if (sb.Length > 0) tokens.Add(sb.ToString(0, sb.Length));
                return tokens;

            }
            catch (Exception)
            {
                return null;
            }
        }


    }
}
