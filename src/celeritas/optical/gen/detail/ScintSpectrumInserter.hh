//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/MatScintSpecInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <variant>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/grid/VectorUtils.hh"
#include "corecel/math/PdfUtils.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/NonuniformGridInserter.hh"
#include "celeritas/inp/OpticalPhysics.hh"

#include "../ScintillationData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Build scintillation spectrum data.
 */
class ScintSpectrumInserter
{
  public:
    //!@{
    //! \name Type aliases
    using Data = HostVal<ScintillationData>;
    using SpectrumId = OptMatId;
    //!@}

  public:
    // Construct with data to insert into
    explicit inline ScintSpectrumInserter(Data* data);

    // Add scintillation data for a single material
    SpectrumId operator()(inp::ScintillationMaterial const& mat);

    // Add empty scintillation data for a single material
    SpectrumId operator()();

  private:
    // Index and inserter types for nonuniform grids (use opaque ID for grids)
    CollectionBuilder<ScintSpectrumRecord, MemSpace::host, OptMatId> spectra_;
    DedupeCollectionBuilder<real_type> reals_;
    CollectionBuilder<ScintDistributionRecord> scint_records_;
    using GridId = OpaqueId<NonuniformGridRecord>;
    NonuniformGridInserter<GridId> insert_energy_cdf_;
    CollectionBuilder<NonuniformGridRecord> energy_cdfs_;

    struct SpectrumVisitor;
};

//! Add to spectrum distribution variant
struct ScintSpectrumInserter::SpectrumVisitor
{
    ScintDistributionRecord& scint;
    NonuniformGridInserter<GridId>& insert_energy_cdf;
    inp::SpectrumArgument spectrum_argument;

    void operator()(inp::NormalDistribution const& norm_dist);
    void operator()(inp::Grid const& grid);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
ScintSpectrumInserter::ScintSpectrumInserter(Data* data)
    : spectra_{&data->spectra}
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
auto ScintSpectrumInserter::operator()(inp::ScintillationMaterial const& mat)
    -> SpectrumId
{
    CELER_EXPECT(mat);

    // Calculate total yield across all components
    double total_yield{0};
    for (auto const& comp : mat.components)
    {
        CELER_VALIDATE(comp.yield > 0,
                       << "invalid yield=" << comp.yield
                       << " for scintillation component (should be positive)");
        total_yield += comp.yield;
    }

    std::vector<double> yield_pdf;
    auto const begin_components = scint_records_.size_id();
    for (auto const& comp : mat.components)
    {
        CELER_VALIDATE(comp.rise_time >= 0,
                       << "invalid rise_time=" << comp.rise_time
                       << " (should be nonnegative)");
        CELER_VALIDATE(comp.fall_time > 0,
                       << "invalid fall_time=" << comp.fall_time
                       << " (should be positive)");

        ScintDistributionRecord scint;
        scint.rise_time = comp.rise_time;
        scint.fall_time = comp.fall_time;

        std::visit(
            SpectrumVisitor{scint, insert_energy_cdf_, comp.spectrum_argument},
            comp.spectrum_distribution);

        scint_records_.push_back(scint);
        yield_pdf.push_back(comp.yield);
    }

    // Normalize yield PDF by total yield
    for (auto& y : yield_pdf)
    {
        y /= total_yield;
    }

    ScintSpectrumRecord spectrum;
    spectrum.yield_per_energy = total_yield;
    spectrum.components = {begin_components, scint_records_.size_id()};
    spectrum.yield_pdf = reals_.insert_back(yield_pdf.begin(), yield_pdf.end());

    CELER_ENSURE(spectrum.components.size() == mat.components.size());
    return spectra_.push_back(std::move(spectrum));
}

//---------------------------------------------------------------------------//
/*!
 * Add an empty record for non-scintillating materials.
 */
auto ScintSpectrumInserter::operator()() -> SpectrumId
{
    return spectra_.push_back({});
}

//---------------------------------------------------------------------------//
/*!
 * Add scintillation data for a single material.
 */
void ScintSpectrumInserter::SpectrumVisitor::operator()(
    inp::NormalDistribution const& norm_dist)
{
    // Gaussian distribution
    CELER_VALIDATE(spectrum_argument == inp::SpectrumArgument::wavelength,
                   << "normal distribution scintillation must use "
                      "wavelength");
    CELER_VALIDATE(norm_dist.mean > 0,
                   << "invalid lambda_mean=" << norm_dist.mean
                   << " for scintillation component (should be "
                      "positive)");
    CELER_VALIDATE(norm_dist.stddev > 0,
                   << "invalid lambda_sigma=" << norm_dist.stddev
                   << " (should be positive)");
    scint.lambda_mean = norm_dist.mean;
    scint.lambda_sigma = norm_dist.stddev;
}

void ScintSpectrumInserter::SpectrumVisitor::operator()(inp::Grid const& grid)
{
    // Explicit grid
    CELER_VALIDATE(is_monotonic_increasing(make_span(grid.x)),
                   << "scintillation spectrum energy grid values are "
                      "not monotonically increasing");

    inp::Grid cdf_grid;
    cdf_grid.x = grid.x;
    cdf_grid.y.resize(grid.x.size());

    if (spectrum_argument != inp::SpectrumArgument::energy)
    {
        CELER_NOT_IMPLEMENTED("wavelength-based grid scintillation spectra");
    }

    // Energy-based spectrum: integrate to get CDF
    SegmentIntegrator integrate_emission{TrapezoidSegmentIntegrator{}};
    integrate_emission(
        make_span(grid.x), make_span(grid.y), make_span(cdf_grid.y));
    normalize_cdf(make_span(cdf_grid.y));

    scint.energy_cdf = insert_energy_cdf(cdf_grid);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
