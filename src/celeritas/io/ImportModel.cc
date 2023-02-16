//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportModel.cc
//---------------------------------------------------------------------------//
#include "ImportModel.hh"

#include <algorithm>
#include <initializer_list>

#include "corecel/Assert.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/StringEnumMapper.hh"

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
}  // namespace

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
        "mu_pair_prod",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Whether the model requires microscopic xs data for sampling.
 *
 * TODO: this could be an implementation detail of ImportModelConverter
 */
bool needs_micro_xs(ImportModelClass value)
{
    CELER_EXPECT(value < ImportModelClass::size_);
    static ModelArray<bool> const needs_xs = make_microxs_flag_array();
    return needs_xs[value];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
