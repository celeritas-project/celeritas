//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"

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
    using VecGrid = std::vector<inp::UniformGrid>;
    using EnergyBound = EnumArray<Bound, double>;
    //!@}

    static constexpr auto energy_units{ImportUnits::mev};
    static constexpr auto xs_units{ImportUnits::len_sq};

    EnergyBound energy{};  //!< Energy bounds for the material
    VecGrid micro_xs;  //!< Cross section for each element
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
    double low_energy_limit{0};
    double high_energy_limit{0};

    explicit operator bool() const
    {
        return model_class != ImportModelClass::size_ && !materials.empty()
               && low_energy_limit < high_energy_limit;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store imported data for multiple scattering.
 */
struct ImportMscModel
{
    //!@{
    //! \name Type aliases
    using PdgInt = int;
    //!@}

    PdgInt particle_pdg{0};
    ImportModelClass model_class{ImportModelClass::size_};
    ImportPhysicsTable xs_table;

    explicit operator bool() const
    {
        return particle_pdg != 0 && model_class != ImportModelClass::size_
               && xs_table;
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
