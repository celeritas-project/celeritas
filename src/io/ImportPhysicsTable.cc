//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-"Battelle", "LLC", and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportPhysicsTable.cc
//---------------------------------------------------------------------------//
#include "ImportPhysicsTable.hh"

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the string value for a table type.
 */
const char* to_cstring(ImportProcessType value)
{
    static const char* const strings[] = {"not_defined",
                                          "transportation",
                                          "electromagnetic",
                                          "optical",
                                          "hadronic",
                                          "photolepton_hadron",
                                          "decay",
                                          "general",
                                          "parameterisation",
                                          "user_defined",
                                          "parallel",
                                          "phonon",
                                          "ucn"};
    REQUIRE(static_cast<int>(value) * sizeof(const char*) < sizeof(strings));
    return strings[static_cast<int>(value)];
}

//---------------------------------------------------------------------------//
/*!
 * Enumerator for the available physics processes.
 *
 * This enum was created to safely access the many physics tables imported.
 */
const char* to_cstring(ImportProcess value)
{
    static const char* const strings[] = {"unknown",
                                          "ion_ioni",
                                          "msc",
                                          "h_ioni",
                                          "h_brems",
                                          "h_pair_prod",
                                          "coulomb_scat",
                                          "e_ioni",
                                          "e_brem",
                                          "photoelectric",
                                          "compton",
                                          "conversion",
                                          "rayleigh",
                                          "annihilation",
                                          "mu_ioni",
                                          "mu_brems",
                                          "mu_pair_prod",
                                          "transportation"};
    REQUIRE(static_cast<int>(value) * sizeof(const char*) < sizeof(strings));
    return strings[static_cast<int>(value)];
}

//---------------------------------------------------------------------------//
/*!
 * Enumerator for the available physics models.
 *
 * This enum was created to safely access the many imported physics tables.
 */
const char* to_cstring(ImportModel value)
{
    static const char* const strings[]
        = {"unknown",        "bragg_ion",         "bethe_bloch",
           "urban_msc",      "icru_73_qo",        "wentzel_VI_uni",
           "h_brem",         "h_pair_prod",       "e_coulomb_scattering",
           "bragg",          "moller_bhabha",     "e_brem_sb",
           "e_brem_lpm",     "e_plus_to_gg",      "livermore_photoelectric",
           "klein_nishina",  "bethe_heitler_lpm", "livermore_rayleigh",
           "mu_bethe_bloch", "mu_brem",           "mu_pair_prod"};
    REQUIRE(static_cast<int>(value) * sizeof(const char*) < sizeof(strings));
    return strings[static_cast<int>(value)];
}

//---------------------------------------------------------------------------//
/*!
 * Get the string value for a table type.
 */
const char* to_cstring(ImportTableType value)
{
    static const char* const strings[] = {
        "dedx",
        "dedx_subsec",
        "dedx_unrestricted",
        "ionisation",
        "ionisation_subsec",
        "csda_range",
        "range",
        "secondary_range",
        "inverse_range",
        "lambda",
        "sublambda",
        "lambda_prim",
    };
    REQUIRE(static_cast<int>(value) * sizeof(const char*) < sizeof(strings));
    return strings[static_cast<int>(value)];
}

//---------------------------------------------------------------------------//
/*!
 * Get the string value for units.
 */
const char* to_cstring(ImportUnits value)
{
    static const char* const strings[] = {
        "unitless",
        "1/cm",
        "1/cm-MeV",
        "MeV",
        "cm",
    };
    REQUIRE(static_cast<int>(value) * sizeof(const char*) < sizeof(strings));
    return strings[static_cast<int>(value)];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
