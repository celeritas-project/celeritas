//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/TransformIO.cc
//---------------------------------------------------------------------------//
#include "TransformIO.hh"

#include "corecel/cont/ArrayIO.hh"

#include "Transformation.hh"
#include "Translation.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, NoTransformation const&)
{
    os << "{}";
    return os;
}

std::ostream& operator<<(std::ostream& os, Translation const& tr)
{
    os << '{' << tr.translation() << '}';
    return os;
}

std::ostream& operator<<(std::ostream& os, Transformation const& tr)
{
    os << '{' << tr.rotation() << ", " << tr.translation() << '}';
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
