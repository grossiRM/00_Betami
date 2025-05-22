using System;

using GeoAPI.Geometries;

namespace USGS.Puma.NTS.Geometries
{
    /// <summary>
    /// A lightweight class used to store 3-dimensional, measured 
    /// coordinates(X, Y, Z, M). NTS only supports topology calculations in the 
    /// x-y plane. The Z and M properties are ignored by all NTS routines such 
    /// as the standard comparison functions.
    /// </summary>
    [Serializable]
    public class Coordinate : ICoordinate, ICoordinateM
    {

        private double x = 0.0;
        /// <summary>
        /// X coordinate.
        /// </summary>
        public double X
        {
            get
            {
                return x;
            }
            set
            {
                x = value;
            }
        }

        private double y = 0.0;
        /// <summary>
        /// Y coordinate.
        /// </summary>
        public double Y
        {
            get
            {
                return y;
            }
            set
            {
                y = value;
            }
        }

        private double z = 0.0;
        /// <summary>
        /// Z coordinate.
        /// </summary>
        public double Z
        {
            get
            {
                return z;
            }
            set
            {
                z = value;
            }
        }

        #region Constructors
        /// <summary>
        /// Constructs a <c>Coordinate</c> at (x,y,z).
        /// </summary>
        /// <param name="x">X value.</param>
        /// <param name="y">Y value.</param>
        /// <param name="z">Z value.</param>
        public Coordinate(double x, double y, double z, double m)
        {
            this.x = x;
            this.y = y;
            this.z = z;
            this.M = m;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="z"></param>
        public Coordinate(double x, double y, double z) : this(x, y, z, 0.0) { }
        /// <summary>
        /// Constructs a <c>Coordinate</c> at (x,y,0).
        /// </summary>
        /// <param name="x">X value.</param>
        /// <param name="y">Y value.</param>
        public Coordinate(double x, double y) : this(x, y, 0.0, 0.0) { }
        /// <summary>
        ///  Constructs a <c>Coordinate</c> at (0,0,0).
        /// </summary>
        public Coordinate() : this(0.0, 0.0, 0.0, 0.0) { }
        /// <summary>
        /// Constructs a <c>Coordinate</c> having the same (x,y,z) values as
        /// <c>other</c>.
        /// </summary>
        /// <param name="c"><c>Coordinate</c> to copy.</param>
        public Coordinate(ICoordinate c) : this(c.X, c.Y, c.Z, 0.0) { }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="c"></param>
        public Coordinate(ICoordinateM c) : this(c.X, c.Y, c.Z, c.M) { }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="c"></param>
        public Coordinate(Coordinate c) : this(c.X, c.Y, c.Z, c.M) { }
        #endregion



        /// <summary>
        /// Gets/Sets <c>Coordinate</c>s (x,y,z) values.
        /// </summary>
        public ICoordinate CoordinateValue
        {
            get
            {
                return this as ICoordinate;
            }
            set
            {
                this.X = value.X;
                this.Y = value.Y;
                this.Z = value.Z;
                if (value is ICoordinateM)
                {
                    this.M = (value as ICoordinateM).M;
                }
                else
                {
                    this.M = 0.0;
                }
            }
        }

        /// <summary>
        /// Returns whether the planar projections of the two <c>Coordinate</c>s are equal.
        ///</summary>
        /// <param name="other"><c>Coordinate</c> with which to do the 2D comparison.</param>
        /// <returns>
        /// <c>true</c> if the x- and y-coordinates are equal;
        /// the Z coordinates do not have to be equal.
        /// </returns>
        public bool Equals2D(ICoordinate other)
        {
            return (x == other.X) && (y == other.Y);
        }

        /// <summary>
        /// Returns <c>true</c> if <c>other</c> has the same values for the x and y ordinates.
        /// Since Coordinates are 2.5D, this routine ignores the z value when making the comparison.
        /// </summary>
        /// <param name="other"><c>Coordinate</c> with which to do the comparison.</param>
        /// <returns><c>true</c> if <c>other</c> is a <c>Coordinate</c> with the same values for the x and y ordinates.</returns>
        public override bool Equals(object other)
        {
            if (other == null)
                return false;
            if (!(other is Coordinate))
                return false;
            return Equals((Coordinate) other);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public Boolean Equals(ICoordinate other)
        {
            return Equals2D(other);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj1"></param>
        /// <param name="obj2"></param>
        /// <returns></returns>
        public static bool operator ==(Coordinate obj1, Coordinate obj2)
        {
            return Object.Equals(obj1, obj2);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj1"></param>
        /// <param name="obj2"></param>
        /// <returns></returns>
        public static bool operator !=(Coordinate obj1, Coordinate obj2)
        {
            return !(obj1 == obj2);
        }  

        /// <summary>
        /// Compares this object with the specified object for order.
        /// Since Coordinates are 2.5D, this routine ignores the z value when making the comparison.
        /// Returns
        ///   -1  : this.x lowerthan other.x || ((this.x == other.x) AND (this.y lowerthan other.y))
        ///    0  : this.x == other.x AND this.y = other.y 
        ///    1  : this.x greaterthan other.x || ((this.x == other.x) AND (this.y greaterthan other.y)) 
        /// </summary>
        /// <param name="o"><c>Coordinate</c> with which this <c>Coordinate</c> is being compared.</param>
        /// <returns>
        /// A negative integer, zero, or a positive integer as this <c>Coordinate</c>
        ///         is less than, equal to, or greater than the specified <c>Coordinate</c>.
        /// </returns>
        public int CompareTo(object o)
        {
            ICoordinate other = (ICoordinate) o;
            return CompareTo(other);
        }

        /// <summary>
        /// Compares this object with the specified object for order.
        /// Since Coordinates are 2.5D, this routine ignores the z value when making the comparison.
        /// Returns
        ///   -1  : this.x lowerthan other.x || ((this.x == other.x) AND (this.y lowerthan other.y))
        ///    0  : this.x == other.x AND this.y = other.y 
        ///    1  : this.x greaterthan other.x || ((this.x == other.x) AND (this.y greaterthan other.y)) 
        /// </summary>
        /// <param name="other"><c>Coordinate</c> with which this <c>Coordinate</c> is being compared.</param>
        /// <returns>
        /// A negative integer, zero, or a positive integer as this <c>Coordinate</c>
        ///         is less than, equal to, or greater than the specified <c>Coordinate</c>.
        /// </returns>
        public int CompareTo(ICoordinate other)
        {
            if (x < other.X)
                return -1;
            if (x > other.X)
                return 1;
            if (y < other.Y)
                return -1;
            if (y > other.Y)
                return 1;
            return 0;
        }

        /// <summary>
        /// Returns <c>true</c> if <c>other</c> has the same values for x, y and z.
        /// </summary>
        /// <param name="other"><c>Coordinate</c> with which to do the 3D comparison.</param>
        /// <returns><c>true</c> if <c>other</c> is a <c>Coordinate</c> with the same values for x, y and z.</returns>
        public bool Equals3D(ICoordinate other)
        {
            return  (x == other.X) && (y == other.Y) && 
                ((z == other.Z) || (Double.IsNaN(Z) && Double.IsNaN(other.Z)));
        }

        /// <summary>
        /// Returns a <c>string</c> of the form <I>(x,y,z)</I> .
        /// </summary>
        /// <returns><c>string</c> of the form <I>(x,y,z)</I></returns>
        public override string ToString()
        {
            return "(" + x + ", " + y + ", " + z + ")";
        }

        /// <summary>
        /// Create a new object as copy of this instance.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            return new Coordinate(this.X, this.Y, this.Z, this.M);
        }

        /// <summary>
        /// Returns distance from <c>p</c> coordinate.
        /// </summary>
        /// <param name="p"><c>Coordinate</c> with which to do the distance comparison.</param>
        /// <returns></returns>
        public double Distance(ICoordinate p)
        {
            double dx = x - p.X;
            double dy = y - p.Y;
            return Math.Sqrt(dx * dx + dy * dy);
        }

        /// <summary>
        /// 
        /// </summary>
        public override int GetHashCode()
        {
            int result = 17;            
            result = 37 * result + GetHashCode(X);
            result = 37 * result + GetHashCode(Y);
            return result;
        }

        /// <summary>
        /// Return HashCode.
        /// </summary>
        /// <param name="value">Value from HashCode computation.</param>
        private static int GetHashCode(double value)
        {
            long f = BitConverter.DoubleToInt64Bits(value);
            return (int)(f^(f>>32));
        }


        #region ICoordinateM Members

        private double _M = 0.0;
        /// <summary>
        /// 
        /// </summary>
        public double M
        {
            get { return _M; }
            set { _M = value; }
        }
        /// <summary>
        /// 
        /// </summary>
        public ICoordinateM CoordinateValueM
        {
            get
            {
                return this as ICoordinateM;
            }
            set
            {
                this.X = value.X;
                this.Y = value.Y;
                this.Z = value.Z;
                this.M = value.M;
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool EqualsM(ICoordinateM other)
        {
            return (this.M == other.M);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals3DM(ICoordinateM other)
        {
            return ((this.Equals3D(other)) && (this.EqualsM(other)));
        }

        #endregion

        #region Overloaded operators added by monoGIS team
        /* BEGIN ADDED BY MPAUL42: monoGIS team */

        /// <summary>
        /// Overloaded + operator.
        /// </summary>
        public static Coordinate operator +(Coordinate coord1, Coordinate coord2)
        {
            return new Coordinate(coord1.X + coord2.X, coord1.Y + coord2.Y, coord1.Z + coord2.Z);
        }

        /// <summary>
        /// Overloaded + operator.
        /// </summary>
        public static Coordinate operator +(Coordinate coord1, double d)
        {
            return new Coordinate(coord1.X + d, coord1.Y + d, coord1.Z + d);
        }

        /// <summary>
        /// Overloaded + operator.
        /// </summary>
        public static Coordinate operator +(double d, Coordinate coord1)
        {
            return coord1 + d;
        }

        /// <summary>
        /// Overloaded * operator.
        /// </summary>
        public static Coordinate operator *(Coordinate coord1, Coordinate coord2)
        {
            return new Coordinate(coord1.X * coord2.X, coord1.Y * coord2.Y, coord1.Z * coord2.Z);
        }

        /// <summary>
        /// Overloaded * operator.
        /// </summary>
        public static Coordinate operator *(Coordinate coord1, double d)
        {
            return new Coordinate(coord1.X * d, coord1.Y * d, coord1.Z * d);
        }

        /// <summary>
        /// Overloaded * operator.
        /// </summary>
        public static Coordinate operator *(double d, Coordinate coord1)
        {
            return coord1 * d;
        }

        /// <summary>
        /// Overloaded - operator.
        /// </summary>
        public static Coordinate operator -(Coordinate coord1, Coordinate coord2)
        {
            return new Coordinate(coord1.X - coord2.X, coord1.Y - coord2.Y, coord1.Z - coord2.Z);
        }

        /// <summary>
        /// Overloaded - operator.
        /// </summary>
        public static Coordinate operator -(Coordinate coord1, double d)
        {
            return new Coordinate(coord1.X - d, coord1.Y - d, coord1.Z - d);
        }

        /// <summary>
        /// Overloaded - operator.
        /// </summary>
        public static Coordinate operator -(double d, Coordinate coord1)
        {
            return coord1 - d;
        }

        /// <summary>
        /// Overloaded / operator.
        /// </summary>
        public static Coordinate operator /(Coordinate coord1, Coordinate coord2)
        {
            return new Coordinate(coord1.X / coord2.X, coord1.Y / coord2.Y, coord1.Z / coord2.Z);
        }

        /// <summary>
        /// Overloaded / operator.
        /// </summary>
        public static Coordinate operator /(Coordinate coord1, double d)
        {
            return new Coordinate(coord1.X / d, coord1.Y / d, coord1.Z / d);
        }

        /// <summary>
        /// Overloaded / operator.
        /// </summary>
        public static Coordinate operator /(double d, Coordinate coord1)
        {
            return coord1 / d;
        }
        /* END ADDED BY MPAUL42: monoGIS team */

        #endregion

    }
} 
