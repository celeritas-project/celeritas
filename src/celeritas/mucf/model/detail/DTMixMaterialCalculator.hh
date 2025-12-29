//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/model/detail/DTMixMaterialCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/EnumArray.hh"
#include "celeritas/inp/MucfPhysics.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/mucf/data/DTMixMucfData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Enum for safely accessing hydrogen isoprotologues.
 *
 * Hydrogen isoprotologue molecules are:
 * - Homonuclear: \f$ ^2H \f$, \f$ ^2d \f$, and \f$ ^2t \f$
 * - Heteronuclear: hd, ht, and dt.
 *
 * \note Muon-catalyzed fusion data is only applicable to a material with
 * concentrations in thermodynamic equilibrium. This equilibrium is calculated
 * at model construction from the material temperature and its h, d, and t
 * fractions.
 */
enum class MucfIsoprotologueMolecule
{
    protium_protium,
    protium_deuterium,
    protium_tritium,
    deuterium_tritium,
    tritium_tritium,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Calculate material-dependent quantities for muon-catalyzed fusion.
 *
 * This class calculates all the muCF data that can be cached during model
 * construction.
 * Use its operator bool to store material data into \c DTMixMucfData :
 * \code
   for (auto matid : range(materials.num_materials()))
   {
       auto mat_view = materials.material_view(PhysMatId{matid});
       DTMixMaterialCalculator material_calculator(mat_view);
       if (material_calculator)
       {
           // Valid d-t mixture material; Store data
       }
   }
 * \endcode
 */
class DTMixMaterialCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using CycleTimesArray = EnumArray<MucfMuonicMolecule, Array<real_type, 2>>;
    //!@}

    //! Construct with material data and calculate all quantities
    DTMixMaterialCalculator(MaterialView const& material);

    //! Get mean cycle times
    CycleTimesArray cycle_times() const { return cycle_times_; }

    //! Check if the material is valid for muon-catalyzed fusion
    explicit operator bool() const { return !cycle_times_.empty(); }

  private:
    using LhdArray = EnumArray<MucfMuonicAtom, real_type>;
    using EquilibriumArray = EnumArray<MucfIsoprotologueMolecule, real_type>;
    using AtomicMassNumber = AtomicNumber;

    //// DATA ////

    MaterialView material_;
    LhdArray lhd_densities_;
    EquilibriumArray eq_densities_;
    EnumArray<MucfMuonicAtom, bool> has_isotope_;
    CycleTimesArray cycle_times_;

    //// LOCAL SCALARS ////
    //! \todo Values are the same used by Acceleron and may need revisiting.
    // {
    // Atomic masses
    static constexpr units::AmuMass protium()
    {
        return units::AmuMass{1.007825031898};
    }

    static constexpr units::AmuMass deuterium()
    {
        return units::AmuMass{2.014101777844};
    }

    static constexpr units::AmuMass tritium()
    {
        return units::AmuMass{3.016049281320};
    }
    //}

    // Liquid hydrogen density (LHD) unit [1/cm^3]
    static constexpr auto liquid_hydrogen_density()
    {
        return units::InvCcDensity{4.25e22};
    }

    //// HELPER FUNCTIONS ////

    // Calculate dt mixture densities in units of liquid hydrogen density
    LhdArray calc_lhd_densities();

    // Calculate thermal equilibrium densities
    EquilibriumArray calc_equilibrium_densities();

    // Return muonic atom from given atomic mass number
    MucfMuonicAtom from_mass_number(AtomicMassNumber mass);

    // Calculate mean fusion cycle times for all reactive muonic molecules
    CycleTimesArray calc_cycle_times(ElementView const& element);

    // Calculate mean fusion cycle times for dd muonic molecules
    Array<real_type, 2> calc_dd_cycle();

    // Calculate mean fusion cycle times for dt muonic molecules
    Array<real_type, 2> calc_dt_cycle();

    // Calculate mean fusion cycle times for tt muonic molecules
    Array<real_type, 2> calc_tt_cycle();
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
