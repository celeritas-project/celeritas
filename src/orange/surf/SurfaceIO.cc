//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceIO.cc
//---------------------------------------------------------------------------//
#include "SurfaceIO.hh"

#include <cmath>
#include <ostream>

#include "corecel/cont/Span.hh"
#include "corecel/cont/SpanIO.hh"

#include "CylCentered.hh"
#include "GeneralQuadric.hh"
#include "PlaneAligned.hh"
#include "Sphere.hh"
#include "SphereCentered.hh"

namespace celeritas
{
#define ORANGE_INSTANTIATE_SHAPE_STREAM(SHAPE)                               \
    template std::ostream& operator<<(std::ostream&, const SHAPE<Axis::x>&); \
    template std::ostream& operator<<(std::ostream&, const SHAPE<Axis::y>&); \
    template std::ostream& operator<<(std::ostream&, const SHAPE<Axis::z>&)

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
template<Axis T>
std::ostream& operator<<(std::ostream& os, PlaneAligned<T> const& s)
{
    os << "Plane: " << to_char(T) << '=' << s.position();
    return os;
}

ORANGE_INSTANTIATE_SHAPE_STREAM(PlaneAligned);
//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, Sphere const& s)
{
    os << "Sphere: r=" << std::sqrt(s.radius_sq()) << " at "
       << make_span(s.origin());
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
