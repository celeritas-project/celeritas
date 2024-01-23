//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/TransformIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>

#include "TransformTypeTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//!@{
//! Print transforms to a stream.

std::ostream& operator<<(std::ostream&, NoTransformation const&);
std::ostream& operator<<(std::ostream&, Translation const&);
std::ostream& operator<<(std::ostream&, Transformation const&);

//!@}
//---------------------------------------------------------------------------//
}  // namespace celeritas
