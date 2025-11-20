//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/MatScintSpecInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/grid/VectorUtils.hh"
#include "corecel/math/PdfUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "celeritas/grid/NonuniformGridInserter.hh"
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

    // Index and inserter types for nonuniform grids (use opaque ID for grids)
    //    using GridInserter = NonuniformGridInserter<GridId>;
    CollectionBuilder<MatScintSpectrum, MemSpace::host, MatId> materials_;
    DedupeCollectionBuilder<real_type> reals_;
    CollectionBuilder<ScintRecord> scint_records_;
    using GridId = OpaqueId<NonuniformGridRecord>;
    NonuniformGridInserter<GridId> insert_energy_cdf_;
    CollectionBuilder<NonuniformGridRecord> energy_cdfs_;
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
    , insert_energy_cdf_(&data->reals, &data->energy_cdfs)
    , energy_cdfs_(&data->energy_cdfs)
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
        CELER_VALIDATE(comp.rise_time >= 0,
                       << "invalid rise_time=" << comp.rise_time
                       << " (should be nonnegative)");
        CELER_VALIDATE(comp.fall_time > 0,
                       << "invalid fall_time=" << comp.fall_time
                       << " (should be positive)");
        CELER_VALIDATE(comp.yield_frac > 0,
                       << "invalid yield=" << comp.yield_frac);

        ScintRecord scint;
        scint.rise_time = comp.rise_time;
        scint.fall_time = comp.fall_time;
        if (comp.spectrum)
        {
            CELER_VALIDATE(is_monotonic_increasing(make_span(comp.spectrum.x)),
                           << "scintillation spectrum energy grid values are "
                              "not "
                              "monotonically increasing");
            inp::Grid grid;
            grid.x = comp.spectrum.x;
            grid.y.resize(grid.x.size());
            SegmentIntegrator integrate_emission{TrapezoidSegmentIntegrator{}};

            integrate_emission(make_span(comp.spectrum.x),
                               make_span(comp.spectrum.y),
                               make_span(grid.y));
            normalize_cdf(make_span(grid.y));
            scint.energy_cdf = insert_energy_cdf_(grid);
        }
        else
        {
            CELER_VALIDATE(comp.gauss.lambda_mean > 0,
                           << "invalid lambda_mean=" << comp.gauss.lambda_mean
                           << " for scintillation component (should be "
                              "positive)");
            CELER_VALIDATE(comp.gauss.lambda_sigma > 0,
                           << "invalid lambda_sigma=" << comp.gauss.lambda_sigma
                           << " (should be positive)");
            scint.lambda_mean = comp.gauss.lambda_mean;
            scint.lambda_sigma = comp.gauss.lambda_sigma;
        }
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
