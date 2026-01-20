//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/model/detail/MucfMaterialInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/mucf/Types.hh"
#include "celeritas/mucf/data/DTMixMucfData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class to calculate and insert muCF material-dependent data into
 * \c DTMixMucfData .
 */
class MucfMaterialInserter
{
  public:
    // Construct with muCF host data
    explicit MucfMaterialInserter(HostVal<DTMixMucfData>* host_data);

    // Insert material if it is a valid d-t mixture
    bool operator()(MaterialView const& material);

  private:
    //// DATA ////

    using MoleculeCycles = Array<real_type, 2>;
    using CycleTimesArray = EnumArray<MucfMuonicMolecule, MoleculeCycles>;
    using LhdArray = EnumArray<MucfMuonicAtom, real_type>;
    using EquilibriumArray = EnumArray<MucfIsoprotologueMolecule, real_type>;
    using AtomicMassNumber = AtomicNumber;
    using MucfIsotope = MucfMuonicAtom;
    using IsotopeChecker = EnumArray<MucfIsotope, bool>;

    // DTMixMucfModel host data references populated by operator()
    CollectionBuilder<PhysMatId, MemSpace::host, MuCfMatId> mucfmatid_to_matid_;
    CollectionBuilder<CycleTimesArray, MemSpace::host, MuCfMatId> cycle_times_;
    // Temporary quantities needed for calculating the model data
    LhdArray lhd_densities_;
    EquilibriumArray equilibrium_densities_;

    //// HELPER FUNCTIONS ////

    // Return muonic atom from given atomic mass number
    MucfMuonicAtom from_mass_number(AtomicMassNumber mass);

    // Calculate dt mixture densities in units of liquid hydrogen density
    LhdArray calc_lhd_densities(ElementView const&);

    // Calculate thermal equilibrium densities
    EquilibriumArray calc_equilibrium_densities(ElementView const&);

    // Calculate mean fusion cycle times for all reactive muonic molecules
    CycleTimesArray calc_cycle_times(ElementView const& element,
                                     IsotopeChecker const& has_isotope);

    // Calculate mean fusion cycle times for dd muonic molecules
    Array<real_type, 2> calc_dd_cycle(ElementView const&);

    // Calculate mean fusion cycle times for dt muonic molecules
    Array<real_type, 2> calc_dt_cycle(ElementView const&);

    // Calculate mean fusion cycle times for tt muonic molecules
    Array<real_type, 2> calc_tt_cycle(ElementView const&);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
