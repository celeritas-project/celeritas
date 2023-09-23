//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeParamsOutput.cc
//---------------------------------------------------------------------------//
#include "OrangeParamsOutput.hh"

#include "celeritas_config.h"
#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"

#include "OrangeParams.hh"  // IWYU pragma: keep
#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

// TODO: Tolerance is defined in OrangeTypes but IO is here
#    include "orange/construct/OrangeInputIO.json.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared orange data.
 */
OrangeParamsOutput::OrangeParamsOutput(SPConstOrangeParams orange)
    : orange_(std::move(orange))
{
    CELER_EXPECT(orange_);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void OrangeParamsOutput::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
    using json = nlohmann::json;

    auto obj = json::object();
    auto const& data = orange_->host_ref();

    // Save param scalars
    {
        auto scalars = json::object();
#    define OPO_SAVE_SCALAR(NAME) scalars[#NAME] = data.scalars.NAME;
        OPO_SAVE_SCALAR(max_depth);
        OPO_SAVE_SCALAR(max_faces);
        OPO_SAVE_SCALAR(max_intersections);
        OPO_SAVE_SCALAR(max_logic_depth);
        OPO_SAVE_SCALAR(tol);
#    undef OPO_SAVE_SCALAR
        obj["scalars"] = std::move(scalars);
    }

    // Save sizes
    {
        auto const& data = orange_->host_ref();

        auto sizes = json::object();
#    define OPO_SAVE_SIZE(NAME) sizes[#NAME] = data.NAME.size()
        OPO_SAVE_SIZE(universe_types);
        OPO_SAVE_SIZE(universe_indices);
        OPO_SAVE_SIZE(simple_units);
        OPO_SAVE_SIZE(rect_arrays);
        OPO_SAVE_SIZE(transforms);
        OPO_SAVE_SIZE(local_surface_ids);
        OPO_SAVE_SIZE(local_volume_ids);
        OPO_SAVE_SIZE(real_ids);
        OPO_SAVE_SIZE(logic_ints);
        OPO_SAVE_SIZE(reals);
        OPO_SAVE_SIZE(surface_types);
        OPO_SAVE_SIZE(connectivity_records);
        OPO_SAVE_SIZE(volume_records);
        OPO_SAVE_SIZE(daughters);
#    undef OPO_SAVE_SIZE

        // BIH data
        {
            auto bih = json::object();
#    define OPO_SAVE_BIH_SIZE(NAME) bih[#NAME] = data.bih_tree_data.NAME.size()
            OPO_SAVE_BIH_SIZE(bboxes);
            OPO_SAVE_BIH_SIZE(local_volume_ids);
            OPO_SAVE_BIH_SIZE(inner_nodes);
            OPO_SAVE_BIH_SIZE(leaf_nodes);
#    undef OPO_SAVE_BIH_SIZE
            sizes["bih"] = std::move(bih);
        }

        // Universe indexer data
        {
            sizes["universe_indexer"] = {
                {"surfaces", data.universe_indexer_data.surfaces.size()},
                {"volumes", data.universe_indexer_data.surfaces.size()},
            };
        }

        obj["sizes"] = std::move(sizes);
    }

    j->obj = std::move(obj);
#else
    (void)sizeof(j);
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
