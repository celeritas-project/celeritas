//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/VariantSurface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <variant>

#include "detail/AllSurfaces.hh"
#include "detail/VariantSurfaceImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! std::variant class of all surfaces.
using VariantSurface = detail::VariantSurface_t;

//---------------------------------------------------------------------------//
}  // namespace celeritas
