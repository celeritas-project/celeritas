//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ScintillationParams.cc
//---------------------------------------------------------------------------//
#include "ScintillationParams.hh"

#include <algorithm>
#include <numeric>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/SoftEqual.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<ScintillationParams>
ScintillationParams::from_import(ImportData const& data)
{
    CELER_EXPECT(!data.optical.empty());

    if (!std::any_of(
            data.optical.begin(), data.optical.end(), [](auto const& iter) {
                return static_cast<bool>(iter.second.scintillation);
            }))
    {
        // No scintillation data present
        return nullptr;
    }

    Input input;
    for (auto const& mat : data.optical)
    {
        input.data.push_back(mat.second.scintillation);
    }
    return std::make_shared<ScintillationParams>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with scintillation input data.
 */
ScintillationParams::ScintillationParams(Input const& input)
{
    CELER_EXPECT(input.data.size() > 0);
    HostVal<ScintillationData> host_data;

    CollectionBuilder spectra(&host_data.spectra);
    CollectionBuilder components(&host_data.components);
    for (auto const& spec : input.data)
    {
        // Check validity of scintillation data
        auto const& comp_inp = spec.components;
        CELER_ASSERT(!comp_inp.empty());
        std::vector<ScintillationComponent> comp(comp_inp.size());
        real_type norm{0};
        for (auto i : range(comp.size()))
        {
            comp[i].lambda_mean = comp_inp[i].lambda_mean;
            comp[i].lambda_sigma = comp_inp[i].lambda_sigma;
            comp[i].rise_time = comp_inp[i].rise_time;
            comp[i].fall_time = comp_inp[i].fall_time;
            norm += comp_inp[i].yield;
        }
        for (auto i : range(comp.size()))
        {
            comp[i].yield_prob = comp_inp[i].yield / norm;
            CELER_VALIDATE(comp[i].yield_prob > 0,
                           << "invalid yield_prob=" << comp[i].yield_prob
                           << " for scintillation component " << i
                           << " (should be positive)");
            CELER_VALIDATE(comp[i].lambda_mean > 0,
                           << "invalid lambda_mean=" << comp[i].lambda_mean
                           << " for scintillation component " << i
                           << " (should be positive)");
            CELER_VALIDATE(comp[i].lambda_sigma > 0,
                           << "invalid lambda_sigma=" << comp[i].lambda_sigma
                           << " for scintillation component " << i
                           << " (should be positive)");
            CELER_VALIDATE(comp[i].rise_time >= 0,
                           << "invalid rise_time=" << comp[i].rise_time
                           << " for scintillation component " << i
                           << " (should be nonnegative)");
            CELER_VALIDATE(comp[i].fall_time > 0,
                           << "invalid fall_time=" << comp[i].fall_time
                           << " for scintillation component " << i
                           << " (should be positive)");
        }
        ScintillationSpectrum spectrum;
        spectrum.yield = spec.yield;
        CELER_VALIDATE(spectrum.yield > 0,
                       << "invalid yield=" << spectrum.yield
                       << " for scintillation (should be positive)");
        spectrum.resolution_scale = spec.resolution_scale;
        CELER_VALIDATE(
            spectrum.resolution_scale >= 0,
            << "invalid resolution_scale=" << spectrum.resolution_scale
            << " for scintillation (should be nonnegative)");
        spectrum.components = components.insert_back(comp.begin(), comp.end());
        spectra.push_back(spectrum);
    }

    mirror_ = CollectionMirror<ScintillationData>{std::move(host_data)};
    CELER_ENSURE(mirror_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
