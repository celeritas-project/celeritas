//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/VariantSurfaces.hh
//---------------------------------------------------------------------------//
#pragma once

#include <variant>

#include "detail/AllSurfaces.hh"
#include "detail/VariantSurfacesImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! std::variant class of all surfaces.
using VariantSurfaces = detail::VariantSurfaces_t;

//---------------------------------------------------------------------------//
}  // namespace celeritas
