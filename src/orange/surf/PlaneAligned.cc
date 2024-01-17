//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/PlaneAligned.cc
//---------------------------------------------------------------------------//
#include "PlaneAligned.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

template class PlaneAligned<Axis::x>;
template class PlaneAligned<Axis::y>;
template class PlaneAligned<Axis::z>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
