//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>
#include <vector>

#include "celeritas/Constants.hh"

#include "ImportPhysicsTable.hh"
#include "ImportUnits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Enumerator for the available physics models.
 *
 * This enum was created to safely access the many imported physics tables.
 *
 * \todo reorganize by physics list (major) and particle (minor) so that newly
 * supported models are appended cleanly to the end of the list.
 */
enum class ImportModelClass
{
    other,
    bragg_ion,
    bethe_bloch,
    urban_msc,
    icru_73_qo,
    wentzel_vi_uni,
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
    fluo_photoelectric,
    goudsmit_saunderson,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Imported data for one material in a particular model.
 *
 * Microscopic cross-section data are stored in the element-selector physics
 * vector is in length^2. They will not be present for all model types, as some
 * models only do on-the-fly calculation (e.g., photoelectric effect) or don't
 * depend on elemental interactions (e.g., compton scattering). The \c
 * needs_micro_xs function indicates which models should store the cross
 * section data.
 *
 * The energy grid's boundaries determine the model's energy bounds and will
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
    static constexpr auto xs_units{ImportUnits::len_sq};
#endif

    std::vector<double> energy;  //!< Energy grid for the material
    std::vector<MicroXs> micro_xs;  //!< Cross sections for each element

    explicit operator bool() const { return energy.size() >= 2; }
};

//---------------------------------------------------------------------------//
/*!
 * Parameters common to all models.
 */
struct ImportModelParameters
{
    double polar_angle_limit{};

    explicit operator bool() const
    {
        return polar_angle_limit >= 0 && polar_angle_limit <= constants::pi;
    }
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
    ImportModelParameters params;

    explicit operator bool() const
    {
        return model_class != ImportModelClass::size_ && !materials.empty()
               && params;
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
    ImportPhysicsTable xs_table;
    ImportModelParameters params;

    explicit operator bool() const
    {
        return particle_pdg != 0 && model_class != ImportModelClass::size_
               && xs_table && params;
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Get the string form of one of the enumerations
char const* to_cstring(ImportModelClass value);

// Get the default Geant4 process name
char const* to_geant_name(ImportModelClass value);

// Convert a Geant4 process name to an IMC (throw RuntimeError if unsupported)
ImportModelClass geant_name_to_import_model_class(std::string_view s);

//---------------------------------------------------------------------------//
}  // namespace celeritas
