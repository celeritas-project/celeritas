//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>

#include "orange/Types.hh"

#include "SurfaceTypeTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//!@{
//! Print surfaces to a stream.
template<Axis T>
std::ostream& operator<<(std::ostream&, const CylCentered<T>&);

std::ostream& operator<<(std::ostream&, const GeneralQuadric&);

template<Axis T>
std::ostream& operator<<(std::ostream&, const PlaneAligned<T>&);

std::ostream& operator<<(std::ostream&, const Sphere&);

std::ostream& operator<<(std::ostream&, const SphereCentered&);
//!@}
//---------------------------------------------------------------------------//
} // namespace celeritas
