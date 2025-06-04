//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoParamsOutput.cc
//---------------------------------------------------------------------------//
#include "GeoParamsOutput.hh"

#include <utility>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/LabelIO.json.hh"

#include "BoundingBoxIO.json.hh"
#include "GeoParamsInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared geometry data.
 */
GeoParamsOutput::GeoParamsOutput(SPConstGeoParams geo) : geo_(std::move(geo))
{
    CELER_EXPECT(geo_);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void GeoParamsOutput::output(JsonPimpl* j) const
{
    using json = nlohmann::json;

    auto obj = json::object();

    obj["supports_safety"] = geo_->supports_safety();
    obj["bbox"] = geo_->bbox();
    obj["max_depth"] = geo_->max_depth();

    // Save volume names
    {
        auto label = json::array();

        auto const& volumes = geo_->volumes();
        for (auto id : range(VolumeId{volumes.size()}))
        {
            label.push_back(volumes.at(id));
        }
        obj["volumes"] = {
            {"label", std::move(label)},
        };
    }

    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
