//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/MatScintSpecInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/math/SoftEqual.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"

#include "../ScintillationData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Build scintillation spectrum data.
 */
class MatScintSpecInserter
{
  public:
    //!@{
    //! \name Type aliases
    using Data = HostVal<ScintillationData>;
    //!@}

  public:
    // Construct with data to insert into
    explicit inline MatScintSpecInserter(Data* data);

    // Add scintillation data for a single material
    auto operator()(ImportMaterialScintSpectrum const& mat);

  private:
    using MatId = OptMatId;

    CollectionBuilder<MatScintSpectrum, MemSpace::host, MatId> materials_;
    DedupeCollectionBuilder<real_type> reals_;
    CollectionBuilder<ScintRecord> scint_records_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
MatScintSpecInserter::MatScintSpecInserter(Data* data)
    : materials_{&data->materials}
    , reals_{&data->reals}
    , scint_records_{&data->scint_records}
{
    CELER_EXPECT(data);
}

//---------------------------------------------------------------------------//
/*!
 * Add scintillation data for a single material.
 */
auto MatScintSpecInserter::operator()(ImportMaterialScintSpectrum const& mat)
{
    CELER_EXPECT(mat);

    CELER_VALIDATE(mat.yield_per_energy > 0,
                   << "invalid yield=" << mat.yield_per_energy
                   << " for scintillation (should be positive)");

    double total_yield{0};
    std::vector<double> yield_pdf;
    auto const begin_components = scint_records_.size_id();
    for (ImportScintComponent const& comp : mat.components)
    {
        CELER_VALIDATE(comp.lambda_mean > 0,
                       << "invalid lambda_mean=" << comp.lambda_mean
                       << " for scintillation component (should be positive)");
        CELER_VALIDATE(comp.lambda_sigma > 0,
                       << "invalid lambda_sigma=" << comp.lambda_sigma
                       << " (should be positive)");
        CELER_VALIDATE(comp.rise_time >= 0,
                       << "invalid rise_time=" << comp.rise_time
                       << " (should be nonnegative)");
        CELER_VALIDATE(comp.fall_time > 0,
                       << "invalid fall_time=" << comp.fall_time
                       << " (should be positive)");
        CELER_VALIDATE(comp.yield_frac > 0,
                       << "invalid yield=" << comp.yield_frac);

        ScintRecord scint;
        scint.lambda_mean = comp.lambda_mean;
        scint.lambda_sigma = comp.lambda_sigma;
        scint.rise_time = comp.rise_time;
        scint.fall_time = comp.fall_time;
        scint_records_.push_back(scint);

        yield_pdf.push_back(comp.yield_frac);
        total_yield += comp.yield_frac;
    }

    // Normalize yield PDF by total yield
    for (auto& y : yield_pdf)
    {
        y /= total_yield;
    }

    MatScintSpectrum spectrum;
    spectrum.yield_per_energy = mat.yield_per_energy;
    spectrum.components = {begin_components, scint_records_.size_id()};
    spectrum.yield_pdf = reals_.insert_back(yield_pdf.begin(), yield_pdf.end());

    CELER_ENSURE(spectrum.components.size() == mat.components.size());
    return materials_.push_back(std::move(spectrum));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
