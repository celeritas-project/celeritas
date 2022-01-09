//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SurfaceIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include "SurfaceTypeTraits.hh"
#include "../Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//!@{
//! Print surfaces to a stream.
template<Axis T>
std::ostream& operator<<(std::ostream& os, const CylCentered<T>& s);

std::ostream& operator<<(std::ostream& os, const GeneralQuadric& s);

template<Axis T>
std::ostream& operator<<(std::ostream& os, const PlaneAligned<T>& s);

std::ostream& operator<<(std::ostream& os, const Sphere& s);
//!@}
//---------------------------------------------------------------------------//
} // namespace celeritas
