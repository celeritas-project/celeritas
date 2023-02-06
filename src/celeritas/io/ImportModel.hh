//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "ImportPhysicsTable.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Enumerator for the available physics models.
 *
 * This enum was created to safely access the many imported physics tables.
 */
enum class ImportModelClass
{
    other,
    unknown [[deprecated]] = other,
    bragg_ion,
    bethe_bloch,
    urban_msc,
    icru_73_qo,
    wentzel_VI_uni,
    h_brems,
    h_pair_prod,
    e_coulomb_scattering,
    bragg,
    moller_bhabha,
    e_brems_sb,
    e_brems_lpm,
    e_plus_to_gg,
    livermore_photoelectric,
    klein_nishina,
    bethe_heitler,
    bethe_heitler_lpm,
    livermore_rayleigh,
    mu_bethe_bloch,
    mu_brems,
    mu_pair_prod,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Imported data for one material in a particluar model.
 *
 * Microscopic cross-section data are stored in the element-selector physics
 * vector is in cm^2. They will not be present for all model types, as some
 * models only do on-the-fly calculation. The \c needs_micro_xs function
 * indicates which models should store the cross section data.
 *
 * The energy grid's boundaries determines the model's energy bounds and will
 * always be set.
 */
struct ImportModelMaterial
{
    //!@{
    //! \name Type aliases
    using MicroXs = std::vector<double>;
    //!@}

#ifndef SWIG
    static constexpr auto energy_units{ImportUnits::mev};
    static constexpr auto xs_units{ImportUnits::cm_2};
#endif

    std::vector<double> energy;  //!< Energy grid for the material
    std::vector<MicroXs> micro_xs;  //!< Cross sections for each element

    explicit operator bool() const { return energy.size() >= 2; }
};

//---------------------------------------------------------------------------//
/*!
 * Imported data for one model of a process.
 *
 * This is always for a particular particle type since we import Processes
 * as being for a particular particle.
 *
 * The materials vector must always be assigned since we want the lower cutoff
 * energy for each model.
 */
struct ImportModel
{
    ImportModelClass model_class{ImportModelClass::size_};
    std::vector<ImportModelMaterial> materials;

    explicit operator bool() const
    {
        return model_class != ImportModelClass::size_ && !materials.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store imported data for multiple scattering.
 */
struct ImportMscModel
{
    int particle_pdg{0};
    ImportModelClass model_class{ImportModelClass::size_};
    ImportPhysicsTable lambda_table;

    explicit operator bool() const
    {
        return particle_pdg != 0 && model_class != ImportModelClass::size_
               && lambda_table;
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

char const* to_cstring(ImportModelClass value);
// Whether Celeritas requires microscopic xs data for sampling
bool needs_micro_xs(ImportModelClass model);

//---------------------------------------------------------------------------//
}  // namespace celeritas
