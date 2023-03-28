//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantPhysicsOptionsIO.json.cc
//---------------------------------------------------------------------------//
#include "GeantPhysicsOptionsIO.json.hh"

#include <string>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/StringEnumMapper.hh"
#include "corecel/math/QuantityIO.json.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the Bremsstrahlung model selection.
 */
char const* to_cstring(BremsModelSelection value)
{
    static EnumStringMapper<BremsModelSelection> const to_cstring_impl{
        "seltzer_berger",
        "relativistic",
        "all",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the multiple scattering model selection.
 */
char const* to_cstring(MscModelSelection value)
{
    static EnumStringMapper<MscModelSelection> const to_cstring_impl{
        "none",
        "urban",
        "wentzel_vi",
        "all",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the atomic relaxation option.
 */
char const* to_cstring(RelaxationSelection value)
{
    static EnumStringMapper<RelaxationSelection> const to_cstring_impl{
        "none",
        "radiative",
        "all",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
// JSON serializers
//---------------------------------------------------------------------------//
void from_json(nlohmann::json const& j, MscModelSelection& value)
{
    static auto const from_string
        = StringEnumMapper<MscModelSelection>::from_cstring_func(to_cstring,
                                                                 "msc model");
    value = from_string(j.get<std::string>());
}

void to_json(nlohmann::json& j, MscModelSelection const& value)
{
    j = std::string{to_cstring(value)};
}

void from_json(nlohmann::json const& j, BremsModelSelection& value)
{
    static auto const from_string
        = StringEnumMapper<BremsModelSelection>::from_cstring_func(
            to_cstring, "brems model");
    value = from_string(j.get<std::string>());
}

void to_json(nlohmann::json& j, BremsModelSelection const& value)
{
    j = std::string{to_cstring(value)};
}

void from_json(nlohmann::json const& j, RelaxationSelection& value)
{
    static auto const from_string
        = StringEnumMapper<RelaxationSelection>::from_cstring_func(
            to_cstring, "atomic relaxation");
    value = from_string(j.get<std::string>());
}

void to_json(nlohmann::json& j, RelaxationSelection const& value)
{
    j = std::string{to_cstring(value)};
}

//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
void from_json(nlohmann::json const& j, GeantPhysicsOptions& options)
{
    options = {};
#define GPO_LOAD_OPTION(NAME)                 \
    do                                        \
    {                                         \
        if (j.count(#NAME))                   \
            j.at(#NAME).get_to(options.NAME); \
    } while (0)
    GPO_LOAD_OPTION(coulomb_scattering);
    GPO_LOAD_OPTION(rayleigh_scattering);
    GPO_LOAD_OPTION(eloss_fluctuation);
    GPO_LOAD_OPTION(lpm);
    GPO_LOAD_OPTION(integral_approach);
    GPO_LOAD_OPTION(gamma_general);
    GPO_LOAD_OPTION(brems);
    GPO_LOAD_OPTION(msc);
    GPO_LOAD_OPTION(relaxation);
    GPO_LOAD_OPTION(em_bins_per_decade);
    GPO_LOAD_OPTION(min_energy);
    GPO_LOAD_OPTION(max_energy);
    GPO_LOAD_OPTION(linear_loss_limit);
    GPO_LOAD_OPTION(lowest_electron_energy);
    GPO_LOAD_OPTION(msc_range_factor);
    GPO_LOAD_OPTION(msc_safety_factor);
    GPO_LOAD_OPTION(msc_lambda_limit);
    GPO_LOAD_OPTION(verbose);
#undef GPO_LOAD_OPTION
}

//---------------------------------------------------------------------------//
/*!
 * Write options to JSON.
 */
void to_json(nlohmann::json& j, GeantPhysicsOptions const& options)
{
    j = nlohmann::json::object();
#define GPO_SAVE_OPTION(NAME) j[#NAME] = options.NAME
    GPO_SAVE_OPTION(coulomb_scattering);
    GPO_SAVE_OPTION(rayleigh_scattering);
    GPO_SAVE_OPTION(eloss_fluctuation);
    GPO_SAVE_OPTION(lpm);
    GPO_SAVE_OPTION(integral_approach);
    GPO_SAVE_OPTION(gamma_general);
    GPO_SAVE_OPTION(brems);
    GPO_SAVE_OPTION(msc);
    GPO_SAVE_OPTION(relaxation);
    GPO_SAVE_OPTION(em_bins_per_decade);
    GPO_SAVE_OPTION(min_energy);
    GPO_SAVE_OPTION(max_energy);
    GPO_SAVE_OPTION(linear_loss_limit);
    GPO_SAVE_OPTION(lowest_electron_energy);
    GPO_SAVE_OPTION(msc_range_factor);
    GPO_SAVE_OPTION(msc_safety_factor);
    GPO_SAVE_OPTION(msc_lambda_limit);
    GPO_SAVE_OPTION(verbose);
#undef GPO_SAVE_OPTION
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
