//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>

#include "geocel/Types.hh"

#include "SurfaceFwd.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//!@{
//! Print surfaces to a stream.
template<Axis T>
std::ostream& operator<<(std::ostream&, ConeAligned<T> const&);

template<Axis T>
std::ostream& operator<<(std::ostream&, CylAligned<T> const&);

template<Axis T>
std::ostream& operator<<(std::ostream&, CylCentered<T> const&);

std::ostream& operator<<(std::ostream&, GeneralQuadric const&);

std::ostream& operator<<(std::ostream&, Plane const&);

template<Axis T>
std::ostream& operator<<(std::ostream&, PlaneAligned<T> const&);

std::ostream& operator<<(std::ostream&, SimpleQuadric const&);

std::ostream& operator<<(std::ostream&, Sphere const&);

std::ostream& operator<<(std::ostream&, SphereCentered const&);

std::ostream& operator<<(std::ostream&, Involute const&);
//!@}
//---------------------------------------------------------------------------//
}  // namespace celeritas
