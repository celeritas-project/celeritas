//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SurfaceIO.cc
//---------------------------------------------------------------------------//
#include "SurfaceIO.hh"

#include <cmath>
#include "base/SpanIO.hh"
#include "CylCentered.hh"
#include "GeneralQuadric.hh"
#include "PlaneAligned.hh"
#include "Sphere.hh"

namespace celeritas
{
#define ORANGE_INSTANTIATE_SHAPE_STREAM(SHAPE)                               \
    template std::ostream& operator<<(std::ostream&, const SHAPE<Axis::x>&); \
    template std::ostream& operator<<(std::ostream&, const SHAPE<Axis::y>&); \
    template std::ostream& operator<<(std::ostream&, const SHAPE<Axis::z>&)

//---------------------------------------------------------------------------//
template<Axis T>
std::ostream& operator<<(std::ostream& os, const CylCentered<T>& s)
{
    os << "Cyl " << to_char(T) << ": r=" << std::sqrt(s.radius_sq());
    return os;
}

ORANGE_INSTANTIATE_SHAPE_STREAM(CylCentered);
//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, const GeneralQuadric& s)
{
    os << "GQuadric: " << s.second() << ' ' << s.cross() << ' ' << s.first()
       << ' ' << s.zeroth();

    return os;
}

//---------------------------------------------------------------------------//
template<Axis T>
std::ostream& operator<<(std::ostream& os, const PlaneAligned<T>& s)
{
    os << "Plane: " << to_char(T) << '=' << s.position();
    return os;
}

ORANGE_INSTANTIATE_SHAPE_STREAM(PlaneAligned);
//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, const Sphere& s)
{
    os << "Sphere: r=" << std::sqrt(s.radius_sq()) << " at "
       << make_span(s.origin());
    return os;
}

//---------------------------------------------------------------------------//
#undef ORANGE_INSTANTIATE_SHAPE_STREAM
} // namespace celeritas
