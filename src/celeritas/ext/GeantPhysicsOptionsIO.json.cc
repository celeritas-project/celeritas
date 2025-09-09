//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantPhysicsOptionsIO.json.cc
//---------------------------------------------------------------------------//
#include "GeantPhysicsOptionsIO.json.hh"

#include <string>

#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringEnumMapper.hh"
#include "corecel/math/QuantityIO.json.hh"

#include "GeantOpticalPhysicsOptionsIO.json.hh"
#include "GeantPhysicsOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
static char const format_str[] = "geant-physics";

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

void from_json(nlohmann::json const& j, MscStepLimitAlgorithm& value)
{
    static auto const from_string
        = StringEnumMapper<MscStepLimitAlgorithm>::from_cstring_func(
            to_cstring, "msc step algorithm");
    value = from_string(j.get<std::string>());
}

void to_json(nlohmann::json& j, MscStepLimitAlgorithm const& value)
{
    j = std::string{to_cstring(value)};
}

void from_json(nlohmann::json const& j, NuclearFormFactorType& value)
{
    static auto const from_string
        = StringEnumMapper<NuclearFormFactorType>::from_cstring_func(
            to_cstring, "form factor");
    value = from_string(j.get<std::string>());
}

void to_json(nlohmann::json& j, NuclearFormFactorType const& value)
{
    j = std::string{to_cstring(value)};
}

void from_json(nlohmann::json const& j, GeantMuonPhysicsOptions& options)
{
#define GMPO_LOAD_OPTION(NAME) CELER_JSON_LOAD_OPTION(j, options, NAME)
    GMPO_LOAD_OPTION(pair_production);
    GMPO_LOAD_OPTION(ionization);
    GMPO_LOAD_OPTION(bremsstrahlung);
    GMPO_LOAD_OPTION(coulomb);
    if (auto iter = j.find("msc"); iter != j.end())
    {
        if (iter->is_boolean())
        {
            CELER_LOG(warning) << "Deprecated msc option type 'boolean': "
                                  "refactor as 'MscModelSelection' string";
            options.msc = iter->get<bool>() ? MscModelSelection::urban
                                            : MscModelSelection::none;
        }
        else
        {
            iter->get_to(options.msc);
        }
    }
#undef GMPO_LOAD_OPTION
}

void to_json(nlohmann::json& j, GeantMuonPhysicsOptions const& inp)
{
    j = {
        CELER_JSON_PAIR(inp, pair_production),
        CELER_JSON_PAIR(inp, ionization),
        CELER_JSON_PAIR(inp, bremsstrahlung),
        CELER_JSON_PAIR(inp, coulomb),
        CELER_JSON_PAIR(inp, msc),
    };
}

//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
void from_json(nlohmann::json const& j, GeantPhysicsOptions& options)
{
#define GPO_LOAD_OPTION(NAME) CELER_JSON_LOAD_OPTION(j, options, NAME)
    check_format(j, format_str);
    check_units(j, format_str);

    GPO_LOAD_OPTION(compton_scattering);
    GPO_LOAD_OPTION(photoelectric);
    GPO_LOAD_OPTION(rayleigh_scattering);
    GPO_LOAD_OPTION(gamma_conversion);
    GPO_LOAD_OPTION(gamma_general);

    GPO_LOAD_OPTION(coulomb_scattering);
    GPO_LOAD_OPTION(ionization);
    GPO_LOAD_OPTION(annihilation);
    GPO_LOAD_OPTION(brems);
    GPO_LOAD_OPTION(seltzer_berger_limit);
    GPO_LOAD_OPTION(msc);
    GPO_LOAD_OPTION(relaxation);

    GPO_LOAD_OPTION(muon);

    GPO_LOAD_OPTION(em_bins_per_decade);
    GPO_LOAD_OPTION(eloss_fluctuation);
    GPO_LOAD_OPTION(lpm);
    GPO_LOAD_OPTION(integral_approach);

    GPO_LOAD_OPTION(min_energy);
    GPO_LOAD_OPTION(max_energy);
    GPO_LOAD_OPTION(linear_loss_limit);
    GPO_LOAD_OPTION(lowest_electron_energy);
    GPO_LOAD_OPTION(lowest_muhad_energy);
    GPO_LOAD_OPTION(apply_cuts);
    GPO_LOAD_OPTION(default_cutoff);

    GPO_LOAD_OPTION(msc_displaced);
    GPO_LOAD_OPTION(msc_muhad_displaced);
    GPO_LOAD_OPTION(msc_range_factor);
    GPO_LOAD_OPTION(msc_muhad_range_factor);
    GPO_LOAD_OPTION(msc_safety_factor);
    GPO_LOAD_OPTION(msc_lambda_limit);
    GPO_LOAD_OPTION(msc_theta_limit);
    GPO_LOAD_OPTION(angle_limit_factor);
    GPO_LOAD_OPTION(msc_step_algorithm);
    GPO_LOAD_OPTION(msc_muhad_step_algorithm);
    GPO_LOAD_OPTION(form_factor);

    GPO_LOAD_OPTION(verbose);

    GPO_LOAD_OPTION(optical);
#undef GPO_LOAD_OPTION
}

//---------------------------------------------------------------------------//
/*!
 * Write options to JSON.
 */
void to_json(nlohmann::json& j, GeantPhysicsOptions const& inp)
{
    j = {
        CELER_JSON_PAIR(inp, compton_scattering),
        CELER_JSON_PAIR(inp, photoelectric),
        CELER_JSON_PAIR(inp, rayleigh_scattering),
        CELER_JSON_PAIR(inp, gamma_conversion),
        CELER_JSON_PAIR(inp, gamma_general),

        CELER_JSON_PAIR(inp, coulomb_scattering),
        CELER_JSON_PAIR(inp, ionization),
        CELER_JSON_PAIR(inp, annihilation),
        CELER_JSON_PAIR(inp, brems),
        CELER_JSON_PAIR(inp, seltzer_berger_limit),
        CELER_JSON_PAIR(inp, msc),
        CELER_JSON_PAIR(inp, relaxation),

        CELER_JSON_PAIR(inp, muon),

        CELER_JSON_PAIR(inp, em_bins_per_decade),
        CELER_JSON_PAIR(inp, eloss_fluctuation),
        CELER_JSON_PAIR(inp, lpm),
        CELER_JSON_PAIR(inp, integral_approach),

        CELER_JSON_PAIR(inp, min_energy),
        CELER_JSON_PAIR(inp, max_energy),
        CELER_JSON_PAIR(inp, linear_loss_limit),
        CELER_JSON_PAIR(inp, lowest_electron_energy),
        CELER_JSON_PAIR(inp, lowest_muhad_energy),
        CELER_JSON_PAIR(inp, apply_cuts),
        CELER_JSON_PAIR(inp, default_cutoff),

        CELER_JSON_PAIR(inp, msc_displaced),
        CELER_JSON_PAIR(inp, msc_muhad_displaced),
        CELER_JSON_PAIR(inp, msc_range_factor),
        CELER_JSON_PAIR(inp, msc_muhad_range_factor),
        CELER_JSON_PAIR(inp, msc_safety_factor),
        CELER_JSON_PAIR(inp, msc_lambda_limit),
        CELER_JSON_PAIR(inp, msc_theta_limit),
        CELER_JSON_PAIR(inp, angle_limit_factor),
        CELER_JSON_PAIR(inp, msc_step_algorithm),
        CELER_JSON_PAIR(inp, msc_muhad_step_algorithm),
        CELER_JSON_PAIR(inp, form_factor),

        CELER_JSON_PAIR(inp, verbose),

        CELER_JSON_PAIR(inp, optical),
    };

    save_format(j, format_str);
    save_units(j);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
