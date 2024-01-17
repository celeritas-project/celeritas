//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportModel.cc
//---------------------------------------------------------------------------//
#include "ImportModel.hh"

#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/StringEnumMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Enumerator for the available physics models.
 *
 * This enum was created to safely access the many imported physics tables.
 */
char const* to_cstring(ImportModelClass value)
{
    static EnumStringMapper<ImportModelClass> const to_cstring_impl{
        "",
        "bragg_ion",
        "bethe_bloch",
        "urban_msc",
        "icru_73_qo",
        "wentzel_vi_uni",
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
        "mu_pair_prod",
        "fluo_photoelectric",
        "goudsmit_saunderson",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get the default Geant4 process name for an ImportProcessClass.
 */
char const* to_geant_name(ImportModelClass value)
{
    static EnumStringMapper<ImportModelClass> const to_name_impl{
        "",
        "BraggIon",
        "BetheBloch",
        "UrbanMsc",
        "ICRU73QO",
        "WentzelVIUni",
        "hBrem",
        "hPairProd",
        "eCoulombScattering",
        "Bragg",
        "MollerBhabha",
        "eBremSB",
        "eBremLPM",
        "eplus2gg",
        "LivermorePhElectric",
        "Klein-Nishina",
        "BetheHeitler",
        "BetheHeitlerLPM",
        "LivermoreRayleigh",
        "MuBetheBloch",
        "MuBrem",
        "muPairProd",
        "PhotoElectric",
        "GoudsmitSaunderson",
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
ImportModelClass geant_name_to_import_model_class(std::string_view s)
{
    static auto const from_string
        = StringEnumMapper<ImportModelClass>::from_cstring_func(to_geant_name,
                                                                "model class");

    return from_string(s);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
