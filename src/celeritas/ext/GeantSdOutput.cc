//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSdOutput.cc
//---------------------------------------------------------------------------//
#include "GeantSdOutput.hh"

#include <G4LogicalVolume.hh>
#include <G4VSensitiveDetector.hh>
#include <nlohmann/json.hpp>

#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/sys/TypeDemangler.hh"

#include "GeantSd.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from hit manager.
 */
GeantSdOutput::GeantSdOutput(SPConstGeantSd hits) : hits_(std::move(hits))
{
    CELER_EXPECT(hits_);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void GeantSdOutput::output(JsonPimpl* j) const
{
    using json = nlohmann::json;

    auto result = json::object();

    // Save detectors
    {
        auto const& celer_vols = hits_->celer_vols();
        auto const& geant_vols = *hits_->geant_vols();
        TypeDemangler<G4VSensitiveDetector> demangle_sd;

        auto vol_ids = json::array();
        auto gv_names = json::array();
        auto sd_names = json::array();
        auto sd_types = json::array();

        for (auto i : range(celer_vols.size()))
        {
            vol_ids.push_back(celer_vols[i].get());

            auto const* lv = geant_vols[i];
            G4VSensitiveDetector const* sd{nullptr};
            if (lv)
            {
                gv_names.push_back(std::string(lv->GetName()));
                sd = lv->GetSensitiveDetector();
            }
            else
            {
                gv_names.push_back(nullptr);
            }

            if (sd)
            {
                sd_names.push_back(std::string(sd->GetName()));
                sd_types.push_back(demangle_sd(*sd));
            }
            else
            {
                sd_names.push_back(nullptr);
                sd_types.push_back(nullptr);
            }
        }

        result["vol_id"] = std::move(vol_ids);
        result["lv_name"] = std::move(gv_names);
        result["sd_name"] = std::move(sd_names);
        result["sd_type"] = std::move(sd_types);
    }

    // TODO: save filter selection
    result["locate_touchable"] = hits_->locate_touchable();

    j->obj = std::move(result);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
