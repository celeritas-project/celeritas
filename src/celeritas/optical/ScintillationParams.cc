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
                return std::any_of(iter.second.begin(),
                                   iter.second.end(),
                                   [](auto const& inner_iter) {
                                       return static_cast<bool>(
                                           inner_iter.second.scintillation);
                                   });
            }))
    {
        // No scintillation data present
        return nullptr;
    }

    Input input;
    for (auto const& part_spectrum : data.optical)
    {
        Input::VecImportScintSpectra mat_spectra;
        for (auto const& mat : part_spectrum.second)
        {
            mat_spectra.push_back(mat.second.scintillation);
        }
        input.data.push_back(std::move(mat_spectra));
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

    CollectionBuilder particle_spectra(&host_data.particle_spectra);
    for (auto const& part_spec : input.data)
    {
        MaterialScintillationData mat_scint_data(part_spec.size());

        for (auto const& mat_spec : part_spec)
        {
            // Check validity of scintillation data
            auto const& comp_inp = mat_spec.material_components;
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
            this->normalize_and_validate(comp, comp_inp, norm);

            ScintillationSpectrum spectrum;
            spectrum.yield = mat_spec.yield;
            CELER_VALIDATE(spectrum.yield > 0,
                           << "invalid yield=" << spectrum.yield
                           << " for scintillation (should be positive)");
            spectrum.resolution_scale = mat_spec.resolution_scale;
            CELER_VALIDATE(
                spectrum.resolution_scale >= 0,
                << "invalid resolution_scale=" << spectrum.resolution_scale
                << " for scintillation (should be nonnegative)");

            mat_scint_data.components.insert_back(comp.begin(), comp.end());
            mat_scint_data.spectra.push_back(spectrum);
        }
        particle_spectra.push_back(mat_scint_data);
    }

    // particle_spectra.insert_back(...)
    mirror_ = CollectionMirror<ScintillationData>{std::move(host_data)};
    CELER_ENSURE(mirror_);
}

//---------------------------------------------------------------------------//
/*
 * Normalize yield probabilities and verify the correctness of the populated
 * ScintillationComponent data.
 */
void ScintillationParams::normalize_and_validate(
    std::vector<ScintillationComponent>& vec_comp,
    std::vector<ImportScintComponent> const& input_comp,
    real_type const norm)
{
    for (auto i : range(vec_comp.size()))
    {
        vec_comp[i].yield_prob = input_comp[i].yield / norm;
        CELER_VALIDATE(vec_comp[i].yield_prob > 0,
                       << "invalid yield_prob=" << vec_comp[i].yield_prob
                       << " for scintillation component " << i
                       << " (should be positive)");
        CELER_VALIDATE(vec_comp[i].lambda_mean > 0,
                       << "invalid lambda_mean=" << vec_comp[i].lambda_mean
                       << " for scintillation component " << i
                       << " (should be positive)");
        CELER_VALIDATE(vec_comp[i].lambda_sigma > 0,
                       << "invalid lambda_sigma=" << vec_comp[i].lambda_sigma
                       << " for scintillation component " << i
                       << " (should be positive)");
        CELER_VALIDATE(vec_comp[i].rise_time >= 0,
                       << "invalid rise_time=" << vec_comp[i].rise_time
                       << " for scintillation component " << i
                       << " (should be nonnegative)");
        CELER_VALIDATE(vec_comp[i].fall_time > 0,
                       << "invalid fall_time=" << vec_comp[i].fall_time
                       << " for scintillation component " << i
                       << " (should be positive)");
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
