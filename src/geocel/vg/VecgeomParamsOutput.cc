//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomParamsOutput.cc
//---------------------------------------------------------------------------//
#include "VecgeomParamsOutput.hh"

#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"

#include "VecgeomParams.hh"  // IWYU pragma: keep

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
    using json = nlohmann::json;

    auto scalars = json::object({
        {"num_volume_levels", vecgeom_->num_volume_levels()},
        {"use_vgdml", vecgeom_->use_vgdml()},
        {"use_surface_tracking", vecgeom_->use_surface_tracking()},
    });
    j->obj = json::object({{"scalars", std::move(scalars)}});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
