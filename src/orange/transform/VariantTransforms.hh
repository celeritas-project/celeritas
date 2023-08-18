//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/VariantTransforms.hh
//---------------------------------------------------------------------------//
#pragma once

#include <variant>

#include "Transformation.hh"
#include "Translation.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! std::variant for all transforms, with optional "no transform"
using VariantTransforms
    = std::variant<std::monostate, Translation, Transformation>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
