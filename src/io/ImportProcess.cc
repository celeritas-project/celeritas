//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-"Battelle", "LLC", and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportProcess.cc
//---------------------------------------------------------------------------//
#include "ImportProcess.hh"

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
    CELER_EXPECT(static_cast<unsigned int>(value) * sizeof(const char*)
                 < sizeof(strings));
    return strings[static_cast<unsigned int>(value)];
}

//---------------------------------------------------------------------------//
/*!
 * Enumerator for the available physics processes.
 *
 * This enum was created to safely access the many physics tables imported.
 */
const char* to_cstring(ImportProcessClass value)
{
    static const char* const strings[] = {"unknown",
                                          "ion_ioni",
                                          "msc",
                                          "h_ioni",
                                          "h_brems",
                                          "h_pair_prod",
                                          "coulomb_scat",
                                          "e_ioni",
                                          "e_brems",
                                          "photoelectric",
                                          "compton",
                                          "conversion",
                                          "rayleigh",
                                          "annihilation",
                                          "mu_ioni",
                                          "mu_brems",
                                          "mu_pair_prod",
                                          "transportation"};
    CELER_EXPECT(static_cast<unsigned int>(value) * sizeof(const char*)
                 < sizeof(strings));
    return strings[static_cast<unsigned int>(value)];
}

//---------------------------------------------------------------------------//
/*!
 * Enumerator for the available physics models.
 *
 * This enum was created to safely access the many imported physics tables.
 */
const char* to_cstring(ImportModelClass value)
{
    static const char* const strings[]
        = {"unknown",        "bragg_ion",         "bethe_bloch",
           "urban_msc",      "icru_73_qo",        "wentzel_VI_uni",
           "h_brems",        "h_pair_prod",       "e_coulomb_scattering",
           "bragg",          "moller_bhabha",     "e_brems_sb",
           "e_brems_lpm",    "e_plus_to_gg",      "livermore_photoelectric",
           "klein_nishina",  "bethe_heitler_lpm", "livermore_rayleigh",
           "mu_bethe_bloch", "mu_brems",          "mu_pair_prod"};
    CELER_EXPECT(static_cast<unsigned int>(value) * sizeof(const char*)
                 < sizeof(strings));
    return strings[static_cast<unsigned int>(value)];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
