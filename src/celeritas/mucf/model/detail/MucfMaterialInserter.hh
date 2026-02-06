//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/model/detail/MucfMaterialInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/inp/MucfPhysics.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/mucf/Types.hh"
#include "celeritas/mucf/data/DTMixMucfData.hh"

#include "EquilibrateDensitiesCalculator.hh"
#include "InterpolatorHelper.hh"

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
    explicit MucfMaterialInserter(HostVal<DTMixMucfData>* host_data,
                                  inp::MucfPhysics const& data);

    // Insert material if it is a valid d-t mixture
    bool operator()(MaterialView const& material);

  private:
    using MoleculeCycles = Array<real_type, 2>;
    using CycleTimesArray = EnumArray<MucfMuonicMolecule, MoleculeCycles>;
    using EquilibriumArray = EquilibrateDensitiesCalculator::EquilibriumArray;
    using MaterialFractionsArray = EnumArray<MucfIsotope, real_type>;
    using AtomicMassNumber = AtomicNumber;
    using InterpolatorsMap
        = std::map<std::pair<inp::CycleTableType, units::HalfSpinInt>,
                   InterpolatorHelper>;

    //// DATA ////

    // DTMixMucfModel host data references populated by operator()
    CollectionBuilder<PhysMatId, MemSpace::host, MuCfMatId> mucfmatid_to_matid_;
    CollectionBuilder<MaterialFractionsArray, MemSpace::host, MuCfMatId>
        isotopic_fractions_;
    CollectionBuilder<CycleTimesArray, MemSpace::host, MuCfMatId> cycle_times_;
    // Const data
    std::map<AtomicMassNumber, MucfIsotope> const mass_isotope_map_{
        {AtomicMassNumber{1}, MucfIsotope::protium},
        {AtomicMassNumber{2}, MucfIsotope::deuterium},
        {AtomicMassNumber{3}, MucfIsotope::tritium},
    };
    inp::MucfPhysics const& data_;
    InterpolatorsMap interpolators_;

    //// HELPER FUNCTIONS ////

    // Calculate mean fusion cycle times for dd muonic molecules
    Array<real_type, 2> calc_dd_cycle(EquilibriumArray const& eq_dens,
                                      real_type const temperature);

    // Calculate mean fusion cycle times for dt muonic molecules
    Array<real_type, 2> calc_dt_cycle(EquilibriumArray const& eq_dens,
                                      real_type const temperature);

    // Calculate mean fusion cycle times for tt muonic molecules
    Array<real_type, 2> calc_tt_cycle(EquilibriumArray const& eq_dens,
                                      real_type const temperature);

    // Get interpolator for given cycle type and spin
    InterpolatorHelper const&
    interpolator(inp::CycleTableType type, units::HalfSpinInt spin) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
