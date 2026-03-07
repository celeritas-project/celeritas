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
    auto operator()(inp::ScintillationMaterial const& mat);

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
auto MatScintSpecInserter::operator()(inp::ScintillationMaterial const& mat)
{
    if (mat.components.empty())
    {
        // No scintillation in this material: add default entry
        MatScintSpectrum spectrum;
        spectrum.yield_per_energy = 0;
        spectrum.components
            = {scint_records_.size_id(), scint_records_.size_id()};
        spectrum.yield_pdf = {reals_.size_id(), reals_.size_id()};
        return materials_.push_back(std::move(spectrum));
    }

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

        ScintRecord scint;
        scint.rise_time = comp.rise_time;
        scint.fall_time = comp.fall_time;

        // Handle spectrum distribution variant
        if (auto const* norm_dist = std::get_if<inp::NormalDistribution>(
                &comp.spectrum_distribution))
        {
            // Gaussian distribution
            CELER_VALIDATE(
                comp.spectrum_argument == inp::SpectrumArgument::wavelength,
                << "normal distribution scintillation must use wavelength");
            CELER_VALIDATE(norm_dist->mean > 0,
                           << "invalid lambda_mean=" << norm_dist->mean
                           << " for scintillation component (should be "
                              "positive)");
            CELER_VALIDATE(norm_dist->stddev > 0,
                           << "invalid lambda_sigma=" << norm_dist->stddev
                           << " (should be positive)");
            scint.lambda_mean = norm_dist->mean;
            scint.lambda_sigma = norm_dist->stddev;
        }
        else if (auto const* grid
                 = std::get_if<inp::Grid>(&comp.spectrum_distribution))
        {
            // Explicit grid
            CELER_VALIDATE(is_monotonic_increasing(make_span(grid->x)),
                           << "scintillation spectrum energy grid values are "
                              "not monotonically increasing");

            inp::Grid cdf_grid;
            cdf_grid.x = grid->x;
            cdf_grid.y.resize(grid->x.size());

            if (comp.spectrum_argument == inp::SpectrumArgument::energy)
            {
                // Energy-based spectrum: integrate to get CDF
                SegmentIntegrator integrate_emission{
                    TrapezoidSegmentIntegrator{}};
                integrate_emission(make_span(grid->x),
                                   make_span(grid->y),
                                   make_span(cdf_grid.y));
                normalize_cdf(make_span(cdf_grid.y));
            }
            else
            {
                // Wavelength-based spectrum: convert to energy first
                CELER_VALIDATE(comp.spectrum_argument
                                   == inp::SpectrumArgument::wavelength,
                               << "unknown spectrum argument type");
                // TODO: implement wavelength->energy conversion for grids
                CELER_NOT_IMPLEMENTED(
                    "wavelength-based grid scintillation spectra");
            }

            scint.energy_cdf = insert_energy_cdf_(cdf_grid);
        }
        else
        {
            CELER_VALIDATE(false, << "invalid spectrum distribution variant");
        }

        scint_records_.push_back(scint);
        yield_pdf.push_back(comp.yield);
    }

    // Normalize yield PDF by total yield
    for (auto& y : yield_pdf)
    {
        y /= total_yield;
    }

    MatScintSpectrum spectrum;
    spectrum.yield_per_energy = total_yield;
    spectrum.components = {begin_components, scint_records_.size_id()};
    spectrum.yield_pdf = reals_.insert_back(yield_pdf.begin(), yield_pdf.end());

    CELER_ENSURE(spectrum.components.size() == mat.components.size());
    return materials_.push_back(std::move(spectrum));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
