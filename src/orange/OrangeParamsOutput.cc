//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeParamsOutput.cc
//---------------------------------------------------------------------------//
#include "OrangeParamsOutput.hh"

#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"

#include "OrangeInputIO.json.hh"
#include "OrangeParams.hh"  // IWYU pragma: keep
#include "OrangeTypesIO.json.hh"

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
    using json = nlohmann::json;

    auto obj = json::object();
    auto const& data = orange_->host_ref();

    // Save param scalars
    obj["scalars"] = [&sdata = data.scalars] {
        auto scalars = json::object();
#define OPO_SAVE_SCALAR(NAME) scalars[#NAME] = sdata.NAME;
        OPO_SAVE_SCALAR(max_depth);
        OPO_SAVE_SCALAR(max_faces);
        OPO_SAVE_SCALAR(max_intersections);
        OPO_SAVE_SCALAR(max_logic_depth);
        OPO_SAVE_SCALAR(tol);
#undef OPO_SAVE_SCALAR
        return scalars;
    }();

    // Save sizes
    obj["sizes"] = [&data] {
        auto sizes = json::object();
#define OPO_SAVE_SIZE(NAME) sizes[#NAME] = data.NAME.size()
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
#undef OPO_SAVE_SIZE

        // Save BIH sizes
        sizes["bih"] = [&bihdata = data.bih_tree_data] {
            auto bih = json::object();
#define OPO_SAVE_BIH_SIZE(NAME) bih[#NAME] = bihdata.NAME.size()
            OPO_SAVE_BIH_SIZE(bboxes);
            OPO_SAVE_BIH_SIZE(local_volume_ids);
            OPO_SAVE_BIH_SIZE(inner_nodes);
            OPO_SAVE_BIH_SIZE(leaf_nodes);
#undef OPO_SAVE_BIH_SIZE
            return bih;
        }();

        return sizes;
    }();

    // Save surface names
    obj["surfaces"] = [&surfaces = orange_->surfaces()] {
        auto label = json::array();
        for (auto id : range(ImplSurfaceId{surfaces.size()}))
        {
            label.push_back(to_string(surfaces.at(id)));
        }
        return json::object({{"label", std::move(label)}});
    }();

    //! \todo Make universe metadata accessible from ORANGE, and write it

    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
