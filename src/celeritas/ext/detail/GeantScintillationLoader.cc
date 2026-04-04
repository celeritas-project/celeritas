//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantScintillationLoader.cc
//---------------------------------------------------------------------------//
#include "GeantScintillationLoader.hh"

#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/io/ImportUnits.hh"

#include "GeantMaterialPropertyGetter.hh"
#include "GeantOpticalMatHelper.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to the scintillation process to populate.
 */
GeantScintillationLoader::GeantScintillationLoader(
    inp::ScintillationProcess& process)
    : process_{process}
{
}

//---------------------------------------------------------------------------//
/*!
 * Load scintillation data for one optical material, printing context on error.
 */
void GeantScintillationLoader::operator()(GeantOpticalMatHelper const& helper)
{
    try
    {
        this->load_one(helper);
    }
    catch (std::exception const&)
    {
        CELER_LOG(error) << "Failed to load scintillation for " << helper;
        throw;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Load scintillation data for one optical material.
 *
 * If the material has no scintillation yield the method returns without
 * modifying \c process_ .
 */
void GeantScintillationLoader::load_one(GeantOpticalMatHelper const& helper)
{
    auto const& get_property = helper.property_getter();

    // Load material-wide properties
    double total_yield{0};
    if (!get_property(total_yield, "SCINTILLATIONYIELD", ImportUnits::inv_mev))
    {
        // No scintillation in this material
        // TODO: check that no other properties are present
        return;
    }
    CELER_VALIDATE(total_yield > 0,
                   << "invalid scintillation yield " << total_yield
                   << " [1/MeV]");

    inp::ScintillationMaterial scint_mat;
    get_property(
        scint_mat.resolution_scale, "RESOLUTIONSCALE", ImportUnits::unitless);

    // Loop over scintillation components (up to 3), saving their
    // yields to renormalize based on total yield
    double renorm_yield{0};
    for (int comp_idx : range(1, 4))
    {
        if (auto s = load_spectrum(
                get_property, "SCINTILLATION", std::to_string(comp_idx)))
        {
            renorm_yield += s->yield;
            scint_mat.components.push_back(std::move(*s));
        }
    }

    // Check for deprecated components
    std::vector<inp::ScintillationSpectrum> deprecated_components;
    for (auto prefix : {"FAST", "SLOW"})
    {
        if (auto s = load_spectrum(get_property, prefix, ""))
        {
            deprecated_components.push_back(*s);
        }
    }
    if (!deprecated_components.empty())
    {
        // TODO: could check whether YIELDRATIO is given (2 components) and
        // fill in scintillation data accordingly
        CELER_LOG(warning) << "Ignoring deprecated scintillation "
                              "component properties";
    }

    CELER_VALIDATE(!scint_mat.components.empty(),
                   << "no scintillation components were present (perhaps "
                      "the material has only unsupported particle "
                      "scintillation or legacy G4 fast/slow components?)");

    // Normalize component yields
    for (auto& s : scint_mat.components)
    {
        s.yield *= total_yield / renorm_yield;
    }
    process_.materials.emplace(helper.opt_mat_id(), std::move(scint_mat));
}

//---------------------------------------------------------------------------//
/*!
 * Load a Gaussian wavelength spectrum approximation for a scintillation
 * component.
 *
 * Returns nullopt if neither the deprecated nor the CELER_-prefixed variant of
 * the property is present.
 */
std::optional<inp::NormalDistribution>
GeantScintillationLoader::load_gaussian(GeantMaterialPropertyGetter const& get,
                                        std::string const& prefix,
                                        std::string const& suffix)
{
    inp::NormalDistribution gaussian;
    auto load_gaussian_impl = [&](std::string const& newprefix) {
        auto mean_name = newprefix + "LAMBDAMEAN" + suffix;
        auto sigma_name = newprefix + "LAMBDASIGMA" + suffix;

        bool has_mean = get(gaussian.mean, mean_name, ImportUnits::len);
        bool has_sigma = get(gaussian.stddev, sigma_name, ImportUnits::len);

        CELER_VALIDATE(has_mean == has_sigma,
                       << "incomplete Gaussian spectrum for " << newprefix
                       << suffix << ": both mean and sigma must be present");
        return has_mean;
    };
    // Load, preferring new variable name
    bool has_depr_gaussian = load_gaussian_impl(prefix);
    bool has_gaussian = load_gaussian_impl("CELER_" + prefix);
    if (!has_gaussian && !has_depr_gaussian)
    {
        // Neither is provided
        return std::nullopt;
    }

    if (has_depr_gaussian)
    {
        if (has_gaussian)
        {
            CELER_LOG(warning) << "Ignoring deprecated optical property "
                                  "(missing CELER_ prefix): "
                               << prefix << suffix;
        }
        else
        {
            CELER_LOG(warning)
                << "Omitting CELER_ prefix is deprecated: rename "
                   "optical property to 'CELER_"
                << prefix << suffix << '\'';
        }
    }
    return gaussian;
}

//---------------------------------------------------------------------------//
/*!
 * Load a single scintillation component spectrum.
 *
 * Returns an optional spectrum. If no spectrum data is found, returns nullopt.
 */
std::optional<inp::ScintillationSpectrum>
GeantScintillationLoader::load_spectrum(GeantMaterialPropertyGetter const& get,
                                        std::string const& prefix,
                                        std::string const& suffix)
{
    CELER_EXPECT(get);

    using IU = ImportUnits;

    auto prop = [&prefix, &suffix](char const* base) {
        std::string result = prefix;
        result += base;
        result += suffix;
        return result;
    };

    inp::ScintillationSpectrum s;

    // Component yields are dimensionless relative weights
    bool has_props = false;
    has_props = get(s.yield, prop("YIELD"), IU::unitless) || has_props;
    has_props = get(s.rise_time, prop("RISETIME"), IU::time) || has_props;
    has_props = get(s.fall_time, prop("TIMECONSTANT"), IU::time) || has_props;

    // Load spectrum: explicit grid or Gaussian approximation
    inp::Grid grid;
    bool has_grid = get(grid, prop("COMPONENT"), {IU::mev, IU::unitless});

    auto gaussian = load_gaussian(get, prefix, suffix);
    CELER_VALIDATE(!(has_grid && gaussian),
                   << "conflicting scintillation spectrum definitions for "
                   << prefix + suffix);

    CELER_VALIDATE(has_props == (has_grid || gaussian),
                   << "incomplete spectrum parameters provided");

    if (gaussian)
    {
        s.spectrum_distribution = std::move(*gaussian);
        s.spectrum_argument = inp::SpectrumArgument::wavelength;
    }
    else if (has_grid)
    {
        s.spectrum_distribution = std::move(grid);
        s.spectrum_argument = inp::SpectrumArgument::energy;
    }
    else
    {
        return std::nullopt;
    }

    return s;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
