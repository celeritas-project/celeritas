//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportProcess.cc
//---------------------------------------------------------------------------//
#include "ImportProcess.hh"

#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/StringEnumMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the string value for a table type.
 */
char const* to_cstring(ImportProcessType value)
{
    static EnumStringMapper<ImportProcessType> const to_cstring_impl{
        "",
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
        "ucn",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Enumerator for the available physics processes.
 *
 * This enum was created to safely access the many physics tables imported.
 */
char const* to_cstring(ImportProcessClass value)
{
    static EnumStringMapper<ImportProcessClass> const to_cstring_impl{
        "",
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
        "electro_nuclear",
        "gamma_general",
        "gamma_nuclear",
        "neutron_elastic",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get the default Geant4 process name for an ImportProcessClass.
 */
char const* to_geant_name(ImportProcessClass value)
{
    static EnumStringMapper<ImportProcessClass> const to_name_impl{
        "",  // unknown,
        "ionIoni",  // ion_ioni,
        "msc",  // msc,
        "hIoni",  // h_ioni,
        "hBrems",  // h_brems,
        "hPairProd",  // h_pair_prod,
        "CoulombScat",  // coulomb_scat,
        "eIoni",  // e_ioni,
        "eBrem",  // e_brems,
        "phot",  // photoelectric,
        "compt",  // compton,
        "conv",  // conversion,
        "Rayl",  // rayleigh,
        "annihil",  // annihilation,
        "muIoni",  // mu_ioni,
        "muBrems",  // mu_brems,
        "muPairProd",  // mu_pair_prod,
        "ElectroNuclearProc",  // electro_nuclear,
        "GammaGeneralProc",  // gamma_general,
        "GammaNuclearProc",  // gamma_nuclear,
        "neutronElasticProc",  // neutron_elastic,
    };
    return to_name_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Convert a Geant4 process name to an IPC.
 *
 * This will throw a \c celeritas::RuntimeError if the string is not known to
 * us.
 */
ImportProcessClass geant_name_to_import_process_class(std::string_view s)
{
    static auto const from_string
        = StringEnumMapper<ImportProcessClass>::from_cstring_func(
            to_geant_name, "process class");

    return from_string(s);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
