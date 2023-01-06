//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportProcess.cc
//---------------------------------------------------------------------------//
#include "ImportProcess.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/io/StringEnumMap.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class T>
using ModelArray = EnumArray<ImportModelClass, T>;

namespace
{
//---------------------------------------------------------------------------//
// Return flags for whether microscopic cross sections are needed
ModelArray<bool> make_microxs_flag_array()
{
    ModelArray<bool> result;
    std::fill(result.begin(), result.end(), false);
    for (ImportModelClass c : {
             ImportModelClass::e_brems_sb,
             ImportModelClass::e_brems_lpm,
             ImportModelClass::mu_brems,
             ImportModelClass::mu_pair_prod,
             ImportModelClass::bethe_heitler_lpm,
             ImportModelClass::livermore_rayleigh,
             ImportModelClass::e_coulomb_scattering,
         })
    {
        result[c] = true;
    }
    return result;
}
//---------------------------------------------------------------------------//
} // namespace

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
    static const char* const strings[] = {"unknown",
                                          "bragg_ion",
                                          "bethe_bloch",
                                          "urban_msc",
                                          "icru_73_qo",
                                          "wentzel_VI_uni",
                                          "h_brems",
                                          "h_pair_prod",
                                          "e_coulomb_scattering",
                                          "bragg",
                                          "moller_bhabha",
                                          "e_brems_sb",
                                          "e_brems_lpm",
                                          "e_plus_to_gg",
                                          "livermore_photoelectric",
                                          "klein_nishina",
                                          "bethe_heitler",
                                          "bethe_heitler_lpm",
                                          "livermore_rayleigh",
                                          "mu_bethe_bloch",
                                          "mu_brems",
                                          "mu_pair_prod"};
    CELER_EXPECT(static_cast<unsigned int>(value) * sizeof(const char*)
                 < sizeof(strings));
    return strings[static_cast<unsigned int>(value)];
}

//---------------------------------------------------------------------------//
/*!
 * Get the default Geant4 process name for an ImportProcessClass.
 */
const char* to_geant_name(ImportProcessClass value)
{
    CELER_EXPECT(value != ImportProcessClass::size_);

    static const char* const strings[] = {
        "",            // unknown,
        "ionIoni",     // ion_ioni,
        "msc",         // msc,
        "hIoni",       // h_ioni,
        "hBrems",      // h_brems,
        "hPairProd",   // h_pair_prod,
        "CoulombScat", // coulomb_scat,
        "eIoni",       // e_ioni,
        "eBrem",       // e_brems,
        "phot",        // photoelectric,
        "compt",       // compton,
        "conv",        // conversion,
        "Rayl",        // rayleigh,
        "annihil",     // annihilation,
        "muIoni",      // mu_ioni,
        "muBrems",     // mu_brems,
        "muPairProd",  // mu_pair_prod,
    };
    static_assert(static_cast<unsigned int>(ImportProcessClass::size_)
                          * sizeof(const char*)
                      == sizeof(strings),
                  "Enum strings are incorrect");

    return strings[static_cast<unsigned int>(value)];
};

//---------------------------------------------------------------------------//
/*!
 * Convert a Geant4 process name to an IPC.
 *
 * This will throw a \c celeritas::RuntimeError if the string is not known to
 * us.
 */
ImportProcessClass geant_name_to_import_process_class(const std::string& s)
{
    static const auto from_string
        = StringEnumMap<ImportProcessClass>::from_cstring_func(
            to_geant_name, "process class");

    return from_string(s);
}

//---------------------------------------------------------------------------//
/*!
 * Whether the model requires microscopic xs data for sampling.
 */
bool needs_micro_xs(ImportModelClass value)
{
    CELER_EXPECT(value < ImportModelClass::size_);
    static const ModelArray<bool> needs_xs = make_microxs_flag_array();
    return needs_xs[value];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
