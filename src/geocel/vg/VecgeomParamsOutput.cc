//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomParamsOutput.cc
//---------------------------------------------------------------------------//
#include "VecgeomParamsOutput.hh"

#include "celeritas_config.h"
#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"

#include "VecgeomParams.hh"  // IWYU pragma: keep
#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared vecgeom data.
 */
VecgeomParamsOutput::VecgeomParamsOutput(SPConstVecgeomParams vecgeom)
    : vecgeom_(std::move(vecgeom))
{
    CELER_EXPECT(vecgeom_);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void VecgeomParamsOutput::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
    using json = nlohmann::json;

    auto scalars = json::object({
        {"max_depth", vecgeom_->max_depth()},
        {"use_vgdml", vecgeom_->use_vgdml()},
        {"use_surface_tracking", vecgeom_->use_surface_tracking()},
    });
    j->obj = json::object({{"scalars", std::move(scalars)}});
#else
    (void)sizeof(j);
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
