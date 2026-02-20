//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceIO.cc
//---------------------------------------------------------------------------//
#include "SurfaceIO.hh"

#include <cmath>
#include <ostream>

#include "corecel/cont/Array.hh"
#include "orange/OrangeTypes.hh"

#include "ConeAligned.hh"  // IWYU pragma: associated
#include "CylAligned.hh"  // IWYU pragma: associated
#include "CylCentered.hh"  // IWYU pragma: associated
#include "GeneralQuadric.hh"  // IWYU pragma: associated
#include "Involute.hh"  // IWYU pragma: associated
#include "Plane.hh"  // IWYU pragma: associated
#include "PlaneAligned.hh"  // IWYU pragma: associated
#include "SimpleQuadric.hh"  // IWYU pragma: associated
#include "Sphere.hh"  // IWYU pragma: associated
#include "SphereCentered.hh"  // IWYU pragma: associated

namespace celeritas
{
#define ORANGE_INSTANTIATE_SHAPE_STREAM(SHAPE)                               \
    template std::ostream& operator<<(std::ostream&, const SHAPE<Axis::x>&); \
    template std::ostream& operator<<(std::ostream&, const SHAPE<Axis::y>&); \
    template std::ostream& operator<<(std::ostream&, const SHAPE<Axis::z>&)

//---------------------------------------------------------------------------//
template<Axis T>
std::ostream& operator<<(std::ostream& os, ConeAligned<T> const& s)
{
    os << "Cone " << to_char(T) << ": t=" << std::sqrt(s.tangent_sq())
       << " at " << s.origin();
    return os;
}

ORANGE_INSTANTIATE_SHAPE_STREAM(ConeAligned);

//---------------------------------------------------------------------------//
template<Axis T>
std::ostream& operator<<(std::ostream& os, CylAligned<T> const& s)
{
    os << "Cyl " << to_char(T) << ": r=" << std::sqrt(s.radius_sq()) << " at "
       << to_char(s.u_axis()) << '=' << s.origin_u() << ", "
       << to_char(s.v_axis()) << '=' << s.origin_v();
    return os;
}

ORANGE_INSTANTIATE_SHAPE_STREAM(CylAligned);

//---------------------------------------------------------------------------//
template<Axis T>
std::ostream& operator<<(std::ostream& os, CylCentered<T> const& s)
{
    os << "Cyl " << to_char(T) << ": r=" << std::sqrt(s.radius_sq());
    return os;
}

ORANGE_INSTANTIATE_SHAPE_STREAM(CylCentered);

//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, GeneralQuadric const& s)
{
    os << "GQuadric: " << s.second() << ' ' << s.cross() << ' ' << s.first()
       << ' ' << s.zeroth();

    return os;
}

//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, Involute const& s)
{
    std::string sign{"ccw"};
    real_type a = s.displacement_angle();
    if (s.sign() == Chirality::right)
    {
        sign = "cw";
        a = constants::pi - a;
    }

    return os << "Involute " << sign << ": r=" << s.r_b() << ", a=" << a
              << ", t={" << s.tmin() << ',' << s.tmax()
              << "} at x=" << s.origin()[0] << ", y=" << s.origin()[1];
}

//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, Plane const& s)
{
    os << "Plane: n=" << s.normal() << ", d=" << s.displacement();
    return os;
}

//---------------------------------------------------------------------------//
template<Axis T>
std::ostream& operator<<(std::ostream& os, PlaneAligned<T> const& s)
{
    os << "Plane: " << to_char(T) << '=' << s.position();
    return os;
}

ORANGE_INSTANTIATE_SHAPE_STREAM(PlaneAligned);
//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, SimpleQuadric const& s)
{
    os << "SQuadric: " << s.second() << ' ' << s.first() << ' ' << s.zeroth();
    return os;
}

//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, Sphere const& s)
{
    os << "Sphere: r=" << std::sqrt(s.radius_sq()) << " at " << s.origin();
    return os;
}

//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, SphereCentered const& s)
{
    os << "Sphere: r=" << std::sqrt(s.radius_sq());
    return os;
}

//---------------------------------------------------------------------------//
#undef ORANGE_INSTANTIATE_SHAPE_STREAM
}  // namespace celeritas
